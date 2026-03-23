#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SBATCH_TEMPLATE="${REPO_ROOT}/nibi_model_compare/slurm/fathomnet_2026_zero_shot.sbatch"

usage() {
  cat <<'EOF'
Submit a Slurm job that runs the FathomNet 2026 zero-shot SAM3 + Qwen 3.5 pipeline.

Usage:
  nibi_model_compare/submit_fathomnet_2026_zero_shot.sh [options]

Common options:
  --account <name>                    Default: $ACCOUNT or def-kmoran
  --project-data-root <path>          Default: /project/rpp-kmoran/merileo/data
  --dataset-subdir <name>             Default: fathomnet_2026_kaggle
  --dataset-bundle <path>             Default: <project-data-root>/<dataset-subdir>/bundles/fathomnet_2026_kaggle.tar.zst
  --dataset-metadata <path>           Default: <project-data-root>/<dataset-subdir>/bundles/fathomnet_2026_kaggle.metadata.json
  --dataset-split <train|test>        Default: test
  --venv-path <path>                  Default: <repo>/.venv
  --env-file <path>                   Default: <repo>/.env
  --output-root <path>                Default: /scratch/$USER/sam3/runs/fathomnet_2026_kaggle
  --job-name <name>                   Default: fathomnet_2026_zero_shot
  --dry-run                           Print env + sbatch command only
  -h, --help                          Show help

LLM/vLLM options:
  --model-id <hf-model-id>            Default: Qwen/Qwen3.5-27B
  --model-revision <rev>              Optional HF revision
  --server-port <n>                   Default: 8006
  --tp-size <n>                       Default: 1
  --max-model-len <n>                 Default: 16384
  --max-num-seqs <n>                  Default: 1
  --gpu-memory-utilization <f>        Default: 0.90
  --limit-mm-per-prompt <json>        Default: {"image":3,"video":0}
  --vllm-runtime <auto|venv|apptainer> Default: auto
  --apptainer-image <path>            Optional SIF path
  --vllm-cuda-visible-devices <ids>   Default: 0
  --runner-cuda-visible-devices <ids> Default: 1

Runner options:
  --device <cuda|cpu>                 Default: cuda
  --checkpoint-path <path>            Optional local SAM3 checkpoint
  --prompt-profile <name>             Default: fathomnet_2026
  --max-generations <n>               Default: 10
  --max-completion-tokens <n>         Default: 1024
  --image-detail <low|high>           Default: high
  --max-images-per-request <n>        Default: 3
  --agent-image-max-edge <n>          Default: 768
  --agent-image-min-edge <n>          Default: 384
  --confidence-threshold <f>          Default: 0.0
  --max-images <n>                    Optional cap for smoke tests
  --proposal-iou-threshold <f>        Default: 0.75
  --final-iou-threshold <f>           Default: 0.55
  --crop-context-ratio <f>            Default: 0.2
  --classification-min-confidence <f> Default: 0.2
  --compile-image-model               Compile the SAM3 image model
  --debug                             Keep per-image debug artifacts

Slurm options:
  --partition <name>                  Optional partition
  --gpus-per-node <spec>              Default: h100:2
  --cpus-per-task <n>                 Default: 16
  --mem <spec>                        Default: 128000M
  --time <hh:mm:ss>                   Default: 24:00:00
  --output <path>                     Default: /scratch/$USER/sam3/logs/fathomnet_2026_kaggle/<split>/out/%x-%j.out
  --error <path>                      Default: /scratch/$USER/sam3/logs/fathomnet_2026_kaggle/<split>/err/%x-%j.err
EOF
}

account="${ACCOUNT:-def-kmoran}"
partition=""
project_data_root="/project/rpp-kmoran/merileo/data"
dataset_subdir="fathomnet_2026_kaggle"
dataset_bundle=""
dataset_metadata=""
dataset_split="test"
venv_path="${REPO_ROOT}/.venv"
env_file="${REPO_ROOT}/.env"
output_root="${SCRATCH:-/scratch/$USER}/sam3/runs/fathomnet_2026_kaggle"
job_name="fathomnet_2026_zero_shot"
dry_run=0

model_id="Qwen/Qwen3.5-27B"
model_revision=""
server_port="8006"
tp_size="1"
max_model_len="16384"
max_num_seqs="1"
gpu_memory_utilization="0.90"
limit_mm_per_prompt='{"image":3,"video":0}'
vllm_runtime="auto"
apptainer_image=""
vllm_cuda_visible_devices="0"
runner_cuda_visible_devices="1"

device="cuda"
checkpoint_path=""
prompt_profile="fathomnet_2026"
max_generations="10"
max_completion_tokens="1024"
image_detail="high"
max_images_per_request="3"
agent_image_max_edge="768"
agent_image_min_edge="384"
confidence_threshold="0.0"
max_images=""
proposal_iou_threshold="0.75"
final_iou_threshold="0.55"
crop_context_ratio="0.2"
classification_min_confidence="0.2"
compile_image_model="0"
debug="0"

gpus_per_node="h100:2"
cpus_per_task="16"
mem="128000M"
time_limit="24:00:00"
output_path=""
error_path=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --account) account="$2"; shift 2 ;;
    --partition) partition="$2"; shift 2 ;;
    --project-data-root) project_data_root="$2"; shift 2 ;;
    --dataset-subdir) dataset_subdir="$2"; shift 2 ;;
    --dataset-bundle) dataset_bundle="$2"; shift 2 ;;
    --dataset-metadata) dataset_metadata="$2"; shift 2 ;;
    --dataset-split) dataset_split="$2"; shift 2 ;;
    --venv-path) venv_path="$2"; shift 2 ;;
    --env-file) env_file="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    --job-name) job_name="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;

    --model-id) model_id="$2"; shift 2 ;;
    --model-revision) model_revision="$2"; shift 2 ;;
    --server-port) server_port="$2"; shift 2 ;;
    --tp-size) tp_size="$2"; shift 2 ;;
    --max-model-len) max_model_len="$2"; shift 2 ;;
    --max-num-seqs) max_num_seqs="$2"; shift 2 ;;
    --gpu-memory-utilization) gpu_memory_utilization="$2"; shift 2 ;;
    --limit-mm-per-prompt) limit_mm_per_prompt="$2"; shift 2 ;;
    --vllm-runtime) vllm_runtime="$2"; shift 2 ;;
    --apptainer-image) apptainer_image="$2"; shift 2 ;;
    --vllm-cuda-visible-devices) vllm_cuda_visible_devices="$2"; shift 2 ;;
    --runner-cuda-visible-devices) runner_cuda_visible_devices="$2"; shift 2 ;;

    --device) device="$2"; shift 2 ;;
    --checkpoint-path) checkpoint_path="$2"; shift 2 ;;
    --prompt-profile) prompt_profile="$2"; shift 2 ;;
    --max-generations) max_generations="$2"; shift 2 ;;
    --max-completion-tokens) max_completion_tokens="$2"; shift 2 ;;
    --image-detail) image_detail="$2"; shift 2 ;;
    --max-images-per-request) max_images_per_request="$2"; shift 2 ;;
    --agent-image-max-edge) agent_image_max_edge="$2"; shift 2 ;;
    --agent-image-min-edge) agent_image_min_edge="$2"; shift 2 ;;
    --confidence-threshold) confidence_threshold="$2"; shift 2 ;;
    --max-images) max_images="$2"; shift 2 ;;
    --proposal-iou-threshold) proposal_iou_threshold="$2"; shift 2 ;;
    --final-iou-threshold) final_iou_threshold="$2"; shift 2 ;;
    --crop-context-ratio) crop_context_ratio="$2"; shift 2 ;;
    --classification-min-confidence) classification_min_confidence="$2"; shift 2 ;;
    --compile-image-model) compile_image_model="1"; shift ;;
    --debug) debug="1"; shift ;;

    --gpus-per-node) gpus_per_node="$2"; shift 2 ;;
    --cpus-per-task) cpus_per_task="$2"; shift 2 ;;
    --mem) mem="$2"; shift 2 ;;
    --time) time_limit="$2"; shift 2 ;;
    --output) output_path="$2"; shift 2 ;;
    --error) error_path="$2"; shift 2 ;;

    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "$dataset_split" != "train" && "$dataset_split" != "test" ]]; then
  echo "--dataset-split must be train or test." >&2
  exit 1
fi

if [[ -z "$dataset_bundle" ]]; then
  dataset_bundle="${project_data_root}/${dataset_subdir}/bundles/fathomnet_2026_kaggle.tar.zst"
fi
if [[ -z "$dataset_metadata" ]]; then
  dataset_metadata="${project_data_root}/${dataset_subdir}/bundles/fathomnet_2026_kaggle.metadata.json"
fi

if ! limit_mm_per_prompt="$(python3 - "$limit_mm_per_prompt" <<'PY'
import json
import sys

value = sys.argv[1]
try:
    parsed = json.loads(value)
except Exception as exc:
    raise SystemExit(f"Invalid --limit-mm-per-prompt JSON: {value}\n{exc}")
print(json.dumps(parsed, separators=(",", ":")))
PY
)"; then
  exit 1
fi

if ! limit_mm_per_prompt_b64="$(python3 - "$limit_mm_per_prompt" <<'PY'
import base64
import sys

print(base64.b64encode(sys.argv[1].encode("utf-8")).decode("ascii"))
PY
)"; then
  exit 1
fi

if [[ ! -f "$SBATCH_TEMPLATE" ]]; then
  echo "Missing sbatch template: $SBATCH_TEMPLATE" >&2
  exit 1
fi

if [[ -z "$output_path" ]]; then
  output_path="${SCRATCH:-/scratch/$USER}/sam3/logs/fathomnet_2026_kaggle/${dataset_split}/out/%x-%j.out"
fi
if [[ -z "$error_path" ]]; then
  error_path="${SCRATCH:-/scratch/$USER}/sam3/logs/fathomnet_2026_kaggle/${dataset_split}/err/%x-%j.err"
fi
if [[ "$dry_run" != "1" ]]; then
  mkdir -p "$(dirname "$output_path")" "$(dirname "$error_path")"
fi

sbatch_cmd=(sbatch --parsable)
[[ -n "$limit_mm_per_prompt_b64" ]] && sbatch_cmd+=(--export "ALL,SUBMITTED_LIMIT_MM_PER_PROMPT_B64=$limit_mm_per_prompt_b64")
[[ -n "$account" ]] && sbatch_cmd+=(--account "$account")
[[ -n "$partition" ]] && sbatch_cmd+=(--partition "$partition")
[[ -n "$gpus_per_node" ]] && sbatch_cmd+=(--gpus-per-node "$gpus_per_node")
[[ -n "$cpus_per_task" ]] && sbatch_cmd+=(--cpus-per-task "$cpus_per_task")
[[ -n "$mem" ]] && sbatch_cmd+=(--mem "$mem")
[[ -n "$time_limit" ]] && sbatch_cmd+=(--time "$time_limit")
[[ -n "$job_name" ]] && sbatch_cmd+=(--job-name "$job_name")
[[ -n "$output_path" ]] && sbatch_cmd+=(--output "$output_path")
[[ -n "$error_path" ]] && sbatch_cmd+=(--error "$error_path")
sbatch_cmd+=("$SBATCH_TEMPLATE")

env_vars=(
  "ACCOUNT=$account"
  "REPO_ROOT=$REPO_ROOT"
  "ENV_FILE=$env_file"
  "VENV_PATH=$venv_path"
  "DATASET_BUNDLE=$dataset_bundle"
  "DATASET_METADATA=$dataset_metadata"
  "DATASET_SPLIT=$dataset_split"
  "MODEL_ID=$model_id"
  "MODEL_REVISION=$model_revision"
  "OUT_ROOT=$output_root"
  "SERVER_PORT=$server_port"
  "TP_SIZE=$tp_size"
  "MAX_MODEL_LEN=$max_model_len"
  "MAX_NUM_SEQS=$max_num_seqs"
  "GPU_MEMORY_UTILIZATION=$gpu_memory_utilization"
  "SUBMITTED_LIMIT_MM_PER_PROMPT=$limit_mm_per_prompt"
  "SUBMITTED_LIMIT_MM_PER_PROMPT_B64=$limit_mm_per_prompt_b64"
  "VLLM_RUNTIME=$vllm_runtime"
  "APPTAINER_IMAGE=$apptainer_image"
  "VLLM_CUDA_VISIBLE_DEVICES=$vllm_cuda_visible_devices"
  "RUNNER_CUDA_VISIBLE_DEVICES=$runner_cuda_visible_devices"
  "DEVICE=$device"
  "CHECKPOINT_PATH=$checkpoint_path"
  "PROMPT_PROFILE=$prompt_profile"
  "MAX_GENERATIONS=$max_generations"
  "MAX_COMPLETION_TOKENS=$max_completion_tokens"
  "IMAGE_DETAIL=$image_detail"
  "MAX_IMAGES_PER_REQUEST=$max_images_per_request"
  "AGENT_IMAGE_MAX_EDGE=$agent_image_max_edge"
  "AGENT_IMAGE_MIN_EDGE=$agent_image_min_edge"
  "CONFIDENCE_THRESHOLD=$confidence_threshold"
  "MAX_IMAGES=$max_images"
  "PROPOSAL_IOU_THRESHOLD=$proposal_iou_threshold"
  "FINAL_IOU_THRESHOLD=$final_iou_threshold"
  "CROP_CONTEXT_RATIO=$crop_context_ratio"
  "CLASSIFICATION_MIN_CONFIDENCE=$classification_min_confidence"
  "COMPILE_IMAGE_MODEL=$compile_image_model"
  "DEBUG=$debug"
)

if [[ "$dry_run" == "1" ]]; then
  echo "---- env ----"
  for kv in "${env_vars[@]}"; do
    printf '%q\n' "$kv"
  done
  echo "---- sbatch ----"
  printf '%q ' "${sbatch_cmd[@]}"
  echo
  exit 0
fi

job_id="$(env "${env_vars[@]}" "${sbatch_cmd[@]}")"
echo "Submitted job: ${job_id}"
echo "Bundle: ${dataset_bundle}"
echo "Metadata: ${dataset_metadata}"
echo "Output root: ${output_root}/${dataset_split}"
