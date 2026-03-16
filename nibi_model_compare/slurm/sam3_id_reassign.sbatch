#!/bin/bash
# Template: post-process one or more framewise SAM3 runs and make IDs temporally consistent.

#SBATCH --job-name=sam3_id_reassign
#SBATCH --account=rpp-kmoran
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=96000M
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/sam3/logs/id_reassign/out/%x-%j.out
#SBATCH --error=/scratch/%u/sam3/logs/id_reassign/err/%x-%j.err

set -euo pipefail

ensure_runner_deps() {
  local wheelhouse_args=(
    -f /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v4
    -f /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v3
    -f /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/generic
    -f /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic
  )
  local runner_dep_specs=(
    "opencv-python"
    "numpy"
    "tqdm"
    "openai"
    "pycocotools"
  )
  export PIP_NO_USER=1

  if ! python3 - <<'PY'
import importlib.util
import sys
missing = [
    m for m in ("cv2", "numpy", "tqdm", "openai", "pycocotools")
    if importlib.util.find_spec(m) is None
]
sys.exit(0 if not missing else 1)
PY
  then
    echo "[Info] Installing missing runner deps into active venv..."
    if ! python3 -m pip --isolated install --no-index "${wheelhouse_args[@]}" "${runner_dep_specs[@]}"; then
      PIP_CONFIG_FILE=/dev/null python3 -m pip install -i https://pypi.org/simple "${runner_dep_specs[@]}"
    fi
  fi
}

REPO_ROOT="${REPO_ROOT:-$PWD}"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT}"
DEFAULT_PROJECT_PREFIX="${SCRATCH:-/scratch/$USER}/sam3"
PROJECT_CACHE_ROOT="${PROJECT_CACHE_ROOT:-$DEFAULT_PROJECT_PREFIX/hf-cache}"
LOG_ROOT="${LOG_ROOT:-${SCRATCH:-/scratch/$USER}/sam3/logs/id_reassign/${BATCH_SAFE:-batch}}"
mkdir -p "$LOG_ROOT/out" "$LOG_ROOT/err" "$PROJECT_ROOT" "$PROJECT_CACHE_ROOT"
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

module --force purge
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

if command -v rsync >/dev/null 2>&1; then
  mkdir -p "$PERSISTENT_CACHE_ROOT/hub"
  rsync -a --ignore-existing "$PERSISTENT_CACHE_ROOT/hub/" "$HF_HUB_CACHE/" || true
fi

if [[ -n "${INPUT_DIRS_JSON_B64:-}" ]]; then
  INPUT_DIRS_JSON="$(python3 - "$INPUT_DIRS_JSON_B64" <<'PY'
import base64
import sys

value = sys.argv[1]
print(base64.b64decode(value).decode("utf-8"))
PY
)"
fi
if [[ -z "${INPUT_DIRS_JSON:-}" ]]; then
  echo "INPUT_DIRS_JSON must be set."
  exit 1
fi

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-27B}"
MODEL_REVISION="${MODEL_REVISION:-}"
if [[ -z "$MODEL_REVISION" && "$MODEL_ID" == *:* ]]; then
  MODEL_REVISION="${MODEL_ID#*:}"
  MODEL_ID="${MODEL_ID%%:*}"
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
  echo "Invalid VLLM_RUNTIME '$VLLM_RUNTIME'"
  exit 1
fi

APPTAINER_IMAGE="${APPTAINER_IMAGE:-${SCRATCH:-/scratch/$USER}/vllm-openai-nightly.sif}"
SERVER_PORT="${SERVER_PORT:-8006}"
TP_SIZE="${TP_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
if [[ -n "${SUBMITTED_LIMIT_MM_PER_PROMPT_B64:-}" ]]; then
  LIMIT_MM_PER_PROMPT="$(python3 - "$SUBMITTED_LIMIT_MM_PER_PROMPT_B64" <<'PY'
import base64
import sys

value = sys.argv[1]
print(base64.b64decode(value).decode("utf-8"))
PY
)"
else
  LIMIT_MM_PER_PROMPT="${SUBMITTED_LIMIT_MM_PER_PROMPT:-${LIMIT_MM_PER_PROMPT:-{\"image\":3,\"video\":0}}}"
fi
LIMIT_MM_PER_PROMPT="$(python3 - "$LIMIT_MM_PER_PROMPT" <<'PY'
import json
import sys

value = sys.argv[1]
try:
    parsed = json.loads(value)
except Exception as exc:
    raise SystemExit(f"Invalid LIMIT_MM_PER_PROMPT JSON: {value}\n{exc}")
print(json.dumps(parsed, separators=(",", ":")))
PY
)"
PROMPT_PROFILE="${PROMPT_PROFILE:-underwater}"
PROMPT_PATH="${PROMPT_PATH:-}"
MISSING_MASK_PROMPT_PATH="${MISSING_MASK_PROMPT_PATH:-}"
VERIFY_GAP_FILL_PROMPT_PATH="${VERIFY_GAP_FILL_PROMPT_PATH:-}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-consistent_ids_mllm}"
OUTPUT_VIDEO_NAME="${OUTPUT_VIDEO_NAME:-overlay_consistent_ids.mp4}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-1024}"
MAX_JSON_RETRIES="${MAX_JSON_RETRIES:-2}"
WINDOW_SIZE="${WINDOW_SIZE:-10}"
WINDOW_STRIDE="${WINDOW_STRIDE:-8}"
MAX_GAP_ISSUES_PER_WINDOW="${MAX_GAP_ISSUES_PER_WINDOW:-8}"
GAP_FILL_MAX_ATTEMPTS="${GAP_FILL_MAX_ATTEMPTS:-4}"
GAP_FILL_POINT_CANDIDATES="${GAP_FILL_POINT_CANDIDATES:-6}"
IMAGE_DETAIL="${IMAGE_DETAIL:-high}"
MAX_IMAGES_PER_REQUEST="${MAX_IMAGES_PER_REQUEST:-3}"
IMAGE_MAX_EDGE="${IMAGE_MAX_EDGE:-768}"
IMAGE_MIN_EDGE="${IMAGE_MIN_EDGE:-384}"
COLLAGE_COLS="${COLLAGE_COLS:-2}"
COLLAGE_TILE_MAX_EDGE="${COLLAGE_TILE_MAX_EDGE:-320}"
SAM3_GPU_IDS="${SAM3_GPU_IDS:-0}"
SAM3_IMAGE_SIZE="${SAM3_IMAGE_SIZE:-1008}"
SAM3_OFFLOAD_VIDEO_TO_CPU="${SAM3_OFFLOAD_VIDEO_TO_CPU:-0}"
FILL_MISSING_MASKS="${FILL_MISSING_MASKS:-1}"
RENDER_VIDEO="${RENDER_VIDEO:-1}"
DEBUG="${DEBUG:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
VLLM_CUDA_VISIBLE_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES:-0}"
RUNNER_CUDA_VISIBLE_DEVICES="${RUNNER_CUDA_VISIBLE_DEVICES:-1}"
export SERVER_PORT

echo "=== Launching vLLM ==="
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
  VLLM_CMD+=(--reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}')
fi
if [[ -n "$LIMIT_MM_PER_PROMPT" ]]; then
  VLLM_CMD+=(--limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT")
fi
if [[ "$VLLM_RUNTIME" == "apptainer" ]]; then
  APPTAINERENV_CUDA_VISIBLE_DEVICES="$VLLM_CUDA_VISIBLE_DEVICES" "${VLLM_CMD[@]}" &
else
  CUDA_VISIBLE_DEVICES="$VLLM_CUDA_VISIBLE_DEVICES" "${VLLM_CMD[@]}" &
fi
VLLM_PID=$!

cleanup() {
  kill "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

python3 - <<'PY'
import os
import time
import urllib.request

port = os.environ.get("SERVER_PORT", "8006")
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

mapfile -t INPUT_DIRS < <(python3 - "$INPUT_DIRS_JSON" <<'PY'
import json
import sys

for value in json.loads(sys.argv[1]):
    print(value)
PY
)

RUN_CMD=(
  python3 scripts/reassign_frame_ids_mllm.py
)
for input_dir in "${INPUT_DIRS[@]}"; do
  RUN_CMD+=("$input_dir")
done
RUN_CMD+=(
  --server-url "http://127.0.0.1:${SERVER_PORT}/v1"
  --model "$MODEL_ID"
  --prompt-profile "$PROMPT_PROFILE"
  --output-subdir "$OUTPUT_SUBDIR"
  --output-video-name "$OUTPUT_VIDEO_NAME"
  --max-completion-tokens "$MAX_COMPLETION_TOKENS"
  --max-json-retries "$MAX_JSON_RETRIES"
  --window-size "$WINDOW_SIZE"
  --window-stride "$WINDOW_STRIDE"
  --max-gap-issues-per-window "$MAX_GAP_ISSUES_PER_WINDOW"
  --gap-fill-max-attempts "$GAP_FILL_MAX_ATTEMPTS"
  --gap-fill-point-candidates "$GAP_FILL_POINT_CANDIDATES"
  --image-detail "$IMAGE_DETAIL"
  --max-images-per-request "$MAX_IMAGES_PER_REQUEST"
  --image-max-edge "$IMAGE_MAX_EDGE"
  --image-min-edge "$IMAGE_MIN_EDGE"
  --collage-cols "$COLLAGE_COLS"
  --collage-tile-max-edge "$COLLAGE_TILE_MAX_EDGE"
  --sam3-gpu-ids "$SAM3_GPU_IDS"
  --sam3-image-size "$SAM3_IMAGE_SIZE"
)
if [[ -n "$PROMPT_PATH" ]]; then
  RUN_CMD+=(--prompt-path "$PROMPT_PATH")
fi
if [[ -n "$MISSING_MASK_PROMPT_PATH" ]]; then
  RUN_CMD+=(--missing-mask-prompt-path "$MISSING_MASK_PROMPT_PATH")
fi
if [[ -n "$VERIFY_GAP_FILL_PROMPT_PATH" ]]; then
  RUN_CMD+=(--verify-gap-fill-prompt-path "$VERIFY_GAP_FILL_PROMPT_PATH")
fi
if [[ "$SAM3_OFFLOAD_VIDEO_TO_CPU" == "1" ]]; then
  RUN_CMD+=(--sam3-offload-video-to-cpu)
fi
if [[ "$FILL_MISSING_MASKS" != "1" ]]; then
  RUN_CMD+=(--no-fill-missing-masks)
fi
if [[ "$RENDER_VIDEO" != "1" ]]; then
  RUN_CMD+=(--no-render-video)
fi
if [[ "$DEBUG" == "1" ]]; then
  RUN_CMD+=(--debug)
fi
if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
  RUN_CMD+=(--continue-on-error)
fi

echo "=== Running ID reassignment job ==="
CUDA_VISIBLE_DEVICES="$RUNNER_CUDA_VISIBLE_DEVICES" "${RUN_CMD[@]}"

echo "Finished ID reassignment for ${#INPUT_DIRS[@]} input run(s)."
