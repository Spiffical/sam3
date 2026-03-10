#!/bin/bash
# Template: one video, one per-frame SAM3 agent run, one Slurm job.

#SBATCH --job-name=sam3_agent_frames
#SBATCH --account=rpp-kmoran
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:2
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
    "opencv-python"
  )
  export PIP_NO_USER=1

  if ! python3 - <<'PY'
import importlib.util
import sys
missing = [
    m
    for m in ("timm", "ftfy", "skimage", "sklearn", "pandas", "matplotlib", "cv2")
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

VIDEO_PATH="${VIDEO_PATH:-}"
if [[ -z "$VIDEO_PATH" ]]; then
  echo "VIDEO_PATH must be set."
  exit 1
fi
PROMPT="${PROMPT:-small creatures}"
PROMPT_PROFILE="${PROMPT_PROFILE:-underwater}"
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
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\":1,\"video\":0}}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-1024}"
MAX_GENERATIONS="${MAX_GENERATIONS:-10}"
DEBUG="${DEBUG:-0}"
KEEP_ARTIFACTS="${KEEP_ARTIFACTS:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
MAX_FRAMES="${MAX_FRAMES:-}"
DEVICE="${DEVICE:-cuda}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
SYSTEM_PROMPT_PATH="${SYSTEM_PROMPT_PATH:-}"
ITERATIVE_SYSTEM_PROMPT_PATH="${ITERATIVE_SYSTEM_PROMPT_PATH:-}"
RUNNER_CUDA_VISIBLE_DEVICES="${RUNNER_CUDA_VISIBLE_DEVICES:-1}"
RUNNER_GPU_IDS="${RUNNER_GPU_IDS:-0}"
VLLM_CUDA_VISIBLE_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES:-0}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${SCRATCH:-/scratch/$USER}/sam3/runs/agent_every_frame_${SLURM_JOB_ID}_${RUN_TS}}"
export SERVER_PORT

TMP_BASE="${SLURM_TMPDIR:-/tmp/$USER/slurm_${SLURM_JOB_ID}}"
STAGE_ROOT="${TMP_BASE}/sam3_agent_every_frame"
MODEL_SAFE="${MODEL_ID//\//_}"
STAGE_OUT="${STAGE_ROOT}/${MODEL_SAFE}"
FINAL_OUT="${OUT_ROOT}/${MODEL_SAFE}"

mkdir -p "$STAGE_OUT" "$FINAL_OUT" "$OUT_ROOT"

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
  kill "$VLLM_PID" || true
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
  python3 scripts/run_sam3_agent_every_frame_video.py
  "$VIDEO_PATH"
  --prompt "$PROMPT"
  --prompt-profile "$PROMPT_PROFILE"
  --server-url "http://127.0.0.1:${SERVER_PORT}/v1"
  --model "$MODEL_ID"
  --output-dir "$STAGE_OUT"
  --device "$DEVICE"
  --max-generations "$MAX_GENERATIONS"
  --max-completion-tokens "$MAX_COMPLETION_TOKENS"
)
if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
  RUN_CMD+=(--continue-on-error)
fi
if [[ "$DEBUG" == "1" ]]; then
  RUN_CMD+=(--debug)
fi
if [[ "$KEEP_ARTIFACTS" == "1" ]]; then
  RUN_CMD+=(--keep-artifacts)
fi
if [[ -n "$MAX_FRAMES" ]]; then
  RUN_CMD+=(--max-frames "$MAX_FRAMES")
fi
if [[ -n "$CHECKPOINT_PATH" ]]; then
  RUN_CMD+=(--checkpoint-path "$CHECKPOINT_PATH")
fi
if [[ -n "$SYSTEM_PROMPT_PATH" ]]; then
  RUN_CMD+=(--system-prompt-path "$SYSTEM_PROMPT_PATH")
fi
if [[ -n "$ITERATIVE_SYSTEM_PROMPT_PATH" ]]; then
  RUN_CMD+=(--iterative-system-prompt-path "$ITERATIVE_SYSTEM_PROMPT_PATH")
fi

echo "=== Running per-frame SAM3 agent job ==="
if [[ -n "$RUNNER_CUDA_VISIBLE_DEVICES" ]]; then
  CUDA_VISIBLE_DEVICES="$RUNNER_CUDA_VISIBLE_DEVICES" "${RUN_CMD[@]}"
else
  "${RUN_CMD[@]}"
fi

cp -a "$STAGE_OUT/." "$FINAL_OUT/"

if [[ -f "$FINAL_OUT/summary.json" ]]; then
  python3 - <<PY
import json
import os

summary_path = os.path.join("$FINAL_OUT", "summary.json")
stage_out = os.path.abspath("$STAGE_OUT")
final_out = os.path.abspath("$FINAL_OUT")

with open(summary_path, "r", encoding="utf-8") as f:
    summary = json.load(f)

changed = False
for key in (
    "output_dir",
    "output_video_path",
    "summary_path",
    "frame_results_path",
    "frame_inputs_dir",
    "frame_artifacts_dir",
):
    value = summary.get(key)
    if isinstance(value, str) and value:
        abs_value = os.path.abspath(value)
        if abs_value.startswith(stage_out):
            summary[key] = final_out + abs_value[len(stage_out):]
            changed = True

if changed:
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\\n")
PY
fi

if command -v rsync >/dev/null 2>&1; then
  mkdir -p "$PERSISTENT_CACHE_ROOT/hub"
  rsync -a --ignore-existing "$HF_HUB_CACHE/" "$PERSISTENT_CACHE_ROOT/hub/" || true
fi

cat > "$OUT_ROOT/README.md" <<EOF
# SAM3 Agent Every-Frame Job Output

- Job ID: ${SLURM_JOB_ID}
- Node list: ${SLURM_JOB_NODELIST}
- Model: ${MODEL_ID}
- Video: ${VIDEO_PATH}
- Prompt: ${PROMPT}
- Prompt profile: ${PROMPT_PROFILE}
- Output root: ${OUT_ROOT}
- Model output folder: ${FINAL_OUT}

Files to review first:
- ${FINAL_OUT}/summary.json
- ${FINAL_OUT}/frame_results.jsonl
- ${FINAL_OUT}/overlay.mp4
EOF

if [[ -d "$STAGE_OUT" ]]; then
  safe_delete_dir "$STAGE_OUT"
fi

echo "Final output folder: $FINAL_OUT"
