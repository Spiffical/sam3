#!/bin/bash

#SBATCH --job-name=fathomnet_2026_zero_shot
#SBATCH --account=def-kmoran
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128000M
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/%u/sam3/logs/fathomnet_2026_kaggle/test/out/%x-%j.out
#SBATCH --error=/scratch/%u/sam3/logs/fathomnet_2026_kaggle/test/err/%x-%j.err

set -euo pipefail

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
    "opencv-python"
    "pycocotools"
  )
  export PIP_NO_USER=1

  if ! python3 - <<'PY'
import importlib.util
import sys
missing = [
    m
    for m in ("timm", "ftfy", "skimage", "sklearn", "pandas", "matplotlib", "cv2", "pycocotools")
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
if [[ -z "${DEFAULT_PROJECT_PREFIX:-}" ]]; then
  DEFAULT_PROJECT_PREFIX="${SCRATCH:-/scratch/$USER}/sam3"
fi
PROJECT_CACHE_ROOT="${PROJECT_CACHE_ROOT:-$DEFAULT_PROJECT_PREFIX/hf-cache}"
LOG_ROOT="${LOG_ROOT:-${SCRATCH:-/scratch/$USER}/sam3/logs/fathomnet_2026_kaggle/${DATASET_SPLIT:-test}}"
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

DATASET_BUNDLE="${DATASET_BUNDLE:-}"
DATASET_METADATA="${DATASET_METADATA:-}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
if [[ -z "$DATASET_BUNDLE" || -z "$DATASET_METADATA" ]]; then
  echo "DATASET_BUNDLE and DATASET_METADATA must be set."
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

print(base64.b64decode(sys.argv[1]).decode("utf-8"))
PY
)"
else
  LIMIT_MM_PER_PROMPT="${SUBMITTED_LIMIT_MM_PER_PROMPT:-${LIMIT_MM_PER_PROMPT:-{\"image\":3,\"video\":0}}}"
fi
LIMIT_MM_PER_PROMPT="$(python3 - "$LIMIT_MM_PER_PROMPT" <<'PY'
import json
import sys

parsed = json.loads(sys.argv[1])
print(json.dumps(parsed, separators=(",", ":")))
PY
)"

DEVICE="${DEVICE:-cuda}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
PROMPT_PROFILE="${PROMPT_PROFILE:-fathomnet_2026}"
MAX_GENERATIONS="${MAX_GENERATIONS:-10}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-1024}"
IMAGE_DETAIL="${IMAGE_DETAIL:-high}"
MAX_IMAGES_PER_REQUEST="${MAX_IMAGES_PER_REQUEST:-3}"
AGENT_IMAGE_MAX_EDGE="${AGENT_IMAGE_MAX_EDGE:-768}"
AGENT_IMAGE_MIN_EDGE="${AGENT_IMAGE_MIN_EDGE:-384}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.0}"
MAX_IMAGES="${MAX_IMAGES:-}"
PROPOSAL_IOU_THRESHOLD="${PROPOSAL_IOU_THRESHOLD:-0.75}"
FINAL_IOU_THRESHOLD="${FINAL_IOU_THRESHOLD:-0.55}"
CROP_CONTEXT_RATIO="${CROP_CONTEXT_RATIO:-0.2}"
CLASSIFICATION_MIN_CONFIDENCE="${CLASSIFICATION_MIN_CONFIDENCE:-0.2}"
COMPILE_IMAGE_MODEL="${COMPILE_IMAGE_MODEL:-0}"
DEBUG="${DEBUG:-0}"
RUNNER_CUDA_VISIBLE_DEVICES="${RUNNER_CUDA_VISIBLE_DEVICES:-1}"
VLLM_CUDA_VISIBLE_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES:-0}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"

TMP_BASE="${SLURM_TMPDIR:-/tmp/$USER/slurm_${SLURM_JOB_ID}}"
STAGE_ROOT="${TMP_BASE}/fathomnet_2026_zero_shot"
DATA_STAGE_BASE="${STAGE_ROOT}/dataset"
MODEL_SAFE="${MODEL_ID//\//_}"
RUN_LABEL="${RUN_TS}_job${SLURM_JOB_ID}"
OUT_BASE="${OUT_ROOT:-${SCRATCH:-/scratch/$USER}/sam3/runs/fathomnet_2026_kaggle}"
RUN_ROOT="${OUT_BASE}/${DATASET_SPLIT}/${RUN_LABEL}"
FINAL_OUT="${RUN_ROOT}/${MODEL_SAFE}"
STAGE_OUT="${STAGE_ROOT}/run/${MODEL_SAFE}"

mkdir -p "$DATA_STAGE_BASE" "$STAGE_OUT" "$FINAL_OUT" "$RUN_ROOT" "$OUT_BASE"

echo "=== Staging dataset bundle into temporary storage ==="
STAGING_MANIFEST="${STAGE_OUT}/dataset_staging_manifest.json"
python3 scripts/fathomnet_2026/extract_dataset_bundle.py \
  --bundle-path "$DATASET_BUNDLE" \
  --metadata-path "$DATASET_METADATA" \
  --output-dir "$DATA_STAGE_BASE" \
  --split "$DATASET_SPLIT" \
  --staging-manifest "$STAGING_MANIFEST"

DATASET_JSON_PATH="$(python3 - "$STAGING_MANIFEST" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(payload["dataset_json_path"])
PY
)"
IMAGES_DIR="$(python3 - "$STAGING_MANIFEST" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(payload["images_dir"])
PY
)"

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

RUN_CMD=(
  python3 scripts/fathomnet_2026/run_zero_shot_submission.py
  --dataset-json "$DATASET_JSON_PATH"
  --images-dir "$IMAGES_DIR"
  --output-dir "$STAGE_OUT"
  --server-url "http://127.0.0.1:${SERVER_PORT}/v1"
  --model "$MODEL_ID"
  --device "$DEVICE"
  --prompt-profile "$PROMPT_PROFILE"
  --max-generations "$MAX_GENERATIONS"
  --max-completion-tokens "$MAX_COMPLETION_TOKENS"
  --image-detail "$IMAGE_DETAIL"
  --max-images-per-request "$MAX_IMAGES_PER_REQUEST"
  --agent-image-max-edge "$AGENT_IMAGE_MAX_EDGE"
  --agent-image-min-edge "$AGENT_IMAGE_MIN_EDGE"
  --confidence-threshold "$CONFIDENCE_THRESHOLD"
  --proposal-iou-threshold "$PROPOSAL_IOU_THRESHOLD"
  --final-iou-threshold "$FINAL_IOU_THRESHOLD"
  --crop-context-ratio "$CROP_CONTEXT_RATIO"
  --classification-min-confidence "$CLASSIFICATION_MIN_CONFIDENCE"
)
if [[ -n "$CHECKPOINT_PATH" ]]; then
  RUN_CMD+=(--checkpoint-path "$CHECKPOINT_PATH")
fi
if [[ -n "$MAX_IMAGES" ]]; then
  RUN_CMD+=(--max-images "$MAX_IMAGES")
fi
if [[ "$COMPILE_IMAGE_MODEL" == "1" ]]; then
  RUN_CMD+=(--compile-image-model)
fi
if [[ "$DEBUG" == "1" ]]; then
  RUN_CMD+=(--debug)
fi

echo "=== Running FathomNet 2026 zero-shot submission job ==="
if [[ -n "$RUNNER_CUDA_VISIBLE_DEVICES" ]]; then
  CUDA_VISIBLE_DEVICES="$RUNNER_CUDA_VISIBLE_DEVICES" "${RUN_CMD[@]}"
else
  "${RUN_CMD[@]}"
fi

cp -a "$STAGE_OUT/." "$FINAL_OUT/"

if command -v rsync >/dev/null 2>&1; then
  mkdir -p "$PERSISTENT_CACHE_ROOT/hub"
  rsync -a --ignore-existing "$HF_HUB_CACHE/" "$PERSISTENT_CACHE_ROOT/hub/" || true
fi

cat > "$RUN_ROOT/README.md" <<EOF
# FathomNet 2026 Zero-Shot Job Output

- Job ID: ${SLURM_JOB_ID}
- Node list: ${SLURM_JOB_NODELIST}
- Model: ${MODEL_ID}
- Split: ${DATASET_SPLIT}
- Bundle: ${DATASET_BUNDLE}
- Metadata: ${DATASET_METADATA}
- Extracted dataset json: ${DATASET_JSON_PATH}
- Extracted images dir: ${IMAGES_DIR}
- Final output dir: ${FINAL_OUT}
EOF

echo "=== Job complete ==="
echo "Final output: $FINAL_OUT"
