#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SBATCH_TEMPLATE="${REPO_ROOT}/nibi_model_compare/slurm/sam3_id_reassign.sbatch"

usage() {
  cat <<'EOF'
Submit a Slurm job that runs an MLLM post-process over framewise SAM3 outputs.

Usage:
  nibi_model_compare/submit_sam3_id_reassign.sh [options] --input-dir /path/to/run [--input-dir /path/to/run2 ...]

Required:
  --input-dir <path>                  One or more framewise SAM3 run directories

Common options:
  --account <name>                    Default: $ACCOUNT or rpp-kmoran
  --input-dir <path>                  Input run directory; may be repeated
  --batch-name <name>                 Optional log-group name. Default: first input dir name
  --prompt-profile <name>             Default: underwater
  --prompt-path <path>                Optional override system prompt for reassignment
  --missing-mask-prompt-path <path>   Optional override system prompt for missing-mask detection
  --verify-gap-fill-prompt-path <path> Optional override system prompt for gap-fill verification
  --missed-creatures-prompt-path <path> Optional override system prompt for missed-creature discovery
  --verify-missed-creatures-prompt-path <path> Optional override system prompt for missed-creature verification
  --outlier-mask-prompt-path <path>   Optional override system prompt for outlier-mask review
  --output-subdir <name>              Default: consistent_ids_mllm
  --output-video-name <name>          Default: overlay_consistent_ids.mp4
  --venv-path <path>                  Default: <repo>/.venv
  --env-file <path>                   Default: <repo>/.env
  --job-name <name>                   Default: sam3_id_reassign
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
  --limit-mm-per-prompt <json>        Default: {"image":3,"video":0}; auto-raised to fit enabled stage requests
  --vllm-runtime <auto|venv|apptainer> Default: auto
  --apptainer-image <path>            Optional SIF path
  --vllm-cuda-visible-devices <ids>   Default: 0
  --runner-cuda-visible-devices <ids> Default: 1

Post-process options:
  --stage <name>                      Ordered stage; repeat as needed. Choices: missed_creatures, gap_fill, outlier_filter, id_reassign
  --max-completion-tokens <n>         Default: 1024
  --max-json-retries <n>              Default: 2
  --window-size <n>                   Default: 10
  --window-stride <n>                 Default: 8
  --assignment-history-frames <n>     Default: 8
  --assignment-heuristic-min-score <f> Default: 0.85
  --find-missed-creatures             Enable the missed-creature discovery stage
  --missed-creatures-window-size <n>  Default: 20
  --missed-creatures-window-stride <n> Default: 10
  --max-missed-creature-issues-per-window <n> Default: 4
  --missed-creatures-max-rounds <n>   Default: 10
  --missed-creatures-max-attempts <n> Default: 10
  --missed-creatures-max-images-per-request <n> Default: 20
  --missed-creatures-duplicate-iou-threshold <f> Default: 0.80
  --max-gap-issues-per-window <n>     Default: 8
  --gap-fill-max-attempts <n>         Default: 4
  --gap-fill-point-candidates <n>     Default: 6
  --max-outlier-checks-per-window <n> Default: 12
  --image-detail <low|high>           Default: high
  --max-images-per-request <n>        Default: 3
  --image-max-edge <n>                Default: 768
  --image-min-edge <n>                Default: 384
  --collage-cols <n>                  Default: 2
  --collage-tile-max-edge <n>         Default: 320
  --sam3-gpu-ids <ids>                Default: 0
  --sam3-image-size <n>               Default: 1008
  --sam3-offload-video-to-cpu         Offload SAM3 frames to CPU memory during gap fill
  --no-fill-missing-masks             Disable the gap-fill stage
  --no-filter-outlier-masks           Disable the gap-fill outlier cleanup stage
  --debug                             Keep extra window-level debug artifacts
  --continue-on-error                 Continue to later input dirs if one fails
  --no-render-video                   Skip writing the relabeled overlay video

Slurm options:
  --partition <name>                  Optional partition
  --gpus-per-node <spec>              Default: h100:2
  --cpus-per-task <n>                 Default: 12
  --mem <spec>                        Default: 96000M
  --time <hh:mm:ss>                   Default: 04:00:00
  --output <path>                     Default: /scratch/$USER/sam3/logs/id_reassign/<batch>/out/%x-%j.out
  --error <path>                      Default: /scratch/$USER/sam3/logs/id_reassign/<batch>/err/%x-%j.err
EOF
}

account="${ACCOUNT:-rpp-kmoran}"
partition=""
batch_name=""
prompt_profile="underwater"
prompt_path=""
missing_mask_prompt_path=""
verify_gap_fill_prompt_path=""
missed_creatures_prompt_path=""
verify_missed_creatures_prompt_path=""
outlier_mask_prompt_path=""
output_subdir="consistent_ids_mllm"
output_video_name="overlay_consistent_ids.mp4"
venv_path="${REPO_ROOT}/.venv"
env_file="${REPO_ROOT}/.env"
job_name="sam3_id_reassign"
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

max_completion_tokens="1024"
max_json_retries="2"
window_size="10"
window_stride="8"
assignment_history_frames="8"
assignment_heuristic_min_score="0.85"
missed_creatures_window_size="20"
missed_creatures_window_stride="10"
max_missed_creature_issues_per_window="4"
missed_creatures_max_rounds="10"
missed_creatures_max_attempts="10"
missed_creatures_max_images_per_request="20"
missed_creatures_duplicate_iou_threshold="0.80"
max_gap_issues_per_window="8"
gap_fill_max_attempts="4"
gap_fill_point_candidates="6"
max_outlier_checks_per_window="12"
image_detail="high"
max_images_per_request="3"
image_max_edge="768"
image_min_edge="384"
collage_cols="2"
collage_tile_max_edge="320"
sam3_gpu_ids="0"
sam3_image_size="1008"
sam3_offload_video_to_cpu=0
find_missed_creatures=0
fill_missing_masks=1
filter_outlier_masks=1
render_video=1
debug=0
continue_on_error=0

gpus_per_node="h100:2"
cpus_per_task="12"
mem="96000M"
time_limit="04:00:00"
output_path=""
error_path=""

declare -a input_dirs=()
declare -a stages=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --account) account="$2"; shift 2 ;;
    --partition) partition="$2"; shift 2 ;;
    --input-dir) input_dirs+=("$2"); shift 2 ;;
    --batch-name) batch_name="$2"; shift 2 ;;
    --prompt-profile) prompt_profile="$2"; shift 2 ;;
    --prompt-path) prompt_path="$2"; shift 2 ;;
    --missing-mask-prompt-path) missing_mask_prompt_path="$2"; shift 2 ;;
    --verify-gap-fill-prompt-path) verify_gap_fill_prompt_path="$2"; shift 2 ;;
    --missed-creatures-prompt-path) missed_creatures_prompt_path="$2"; shift 2 ;;
    --verify-missed-creatures-prompt-path) verify_missed_creatures_prompt_path="$2"; shift 2 ;;
    --outlier-mask-prompt-path) outlier_mask_prompt_path="$2"; shift 2 ;;
    --output-subdir) output_subdir="$2"; shift 2 ;;
    --output-video-name) output_video_name="$2"; shift 2 ;;
    --venv-path) venv_path="$2"; shift 2 ;;
    --env-file) env_file="$2"; shift 2 ;;
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

    --stage) stages+=("$2"); shift 2 ;;
    --max-completion-tokens) max_completion_tokens="$2"; shift 2 ;;
    --max-json-retries) max_json_retries="$2"; shift 2 ;;
    --window-size) window_size="$2"; shift 2 ;;
    --window-stride) window_stride="$2"; shift 2 ;;
    --assignment-history-frames) assignment_history_frames="$2"; shift 2 ;;
    --assignment-heuristic-min-score) assignment_heuristic_min_score="$2"; shift 2 ;;
    --find-missed-creatures) find_missed_creatures=1; shift ;;
    --no-find-missed-creatures) find_missed_creatures=0; shift ;;
    --missed-creatures-window-size) missed_creatures_window_size="$2"; shift 2 ;;
    --missed-creatures-window-stride) missed_creatures_window_stride="$2"; shift 2 ;;
    --max-missed-creature-issues-per-window) max_missed_creature_issues_per_window="$2"; shift 2 ;;
    --missed-creatures-max-rounds) missed_creatures_max_rounds="$2"; shift 2 ;;
    --missed-creatures-max-attempts) missed_creatures_max_attempts="$2"; shift 2 ;;
    --missed-creatures-max-images-per-request) missed_creatures_max_images_per_request="$2"; shift 2 ;;
    --missed-creatures-duplicate-iou-threshold) missed_creatures_duplicate_iou_threshold="$2"; shift 2 ;;
    --max-gap-issues-per-window) max_gap_issues_per_window="$2"; shift 2 ;;
    --gap-fill-max-attempts) gap_fill_max_attempts="$2"; shift 2 ;;
    --gap-fill-point-candidates) gap_fill_point_candidates="$2"; shift 2 ;;
    --max-outlier-checks-per-window) max_outlier_checks_per_window="$2"; shift 2 ;;
    --image-detail) image_detail="$2"; shift 2 ;;
    --max-images-per-request) max_images_per_request="$2"; shift 2 ;;
    --image-max-edge) image_max_edge="$2"; shift 2 ;;
    --image-min-edge) image_min_edge="$2"; shift 2 ;;
    --collage-cols) collage_cols="$2"; shift 2 ;;
    --collage-tile-max-edge) collage_tile_max_edge="$2"; shift 2 ;;
    --sam3-gpu-ids) sam3_gpu_ids="$2"; shift 2 ;;
    --sam3-image-size) sam3_image_size="$2"; shift 2 ;;
    --sam3-offload-video-to-cpu) sam3_offload_video_to_cpu=1; shift ;;
    --no-fill-missing-masks) fill_missing_masks=0; shift ;;
    --no-filter-outlier-masks) filter_outlier_masks=0; shift ;;
    --debug) debug=1; shift ;;
    --continue-on-error) continue_on_error=1; shift ;;
    --no-render-video) render_video=0; shift ;;

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

if [[ ${#input_dirs[@]} -eq 0 ]]; then
  echo "At least one --input-dir is required." >&2
  usage
  exit 1
fi

if [[ -z "$batch_name" ]]; then
  if ! batch_name="$(python3 - "${input_dirs[0]}" <<'PY'
from pathlib import Path
import sys

p = Path(sys.argv[1]).expanduser()
name = p.name or "id_reassign"
if p.parent.name == "latest" and p.parent.parent.name:
    name = p.parent.parent.name
elif "_job" in p.parent.name and p.parent.parent.name:
    name = p.parent.parent.name
elif p.parent.name:
    name = p.parent.name
print(name)
PY
)"; then
    exit 1
  fi
fi

if ! batch_safe="$(python3 - "$batch_name" <<'PY'
import re
import sys

safe = re.sub(r"[^A-Za-z0-9._-]+", "_", sys.argv[1]).strip("._-")
print(safe or "id_reassign")
PY
)"; then
  exit 1
fi

if ! input_dirs_json="$(python3 - "${input_dirs[@]}" <<'PY'
import json
import sys

print(json.dumps(sys.argv[1:]))
PY
)"; then
  exit 1
fi

if ! input_dirs_json_b64="$(python3 - "$input_dirs_json" <<'PY'
import base64
import sys

print(base64.b64encode(sys.argv[1].encode("utf-8")).decode("ascii"))
PY
)"; then
  exit 1
fi

if ! limit_mm_per_prompt="$(python3 - \
  "$limit_mm_per_prompt" \
  "$max_images_per_request" \
  "$missed_creatures_max_images_per_request" \
  "$fill_missing_masks" \
  "$find_missed_creatures" <<'PY'
import json
import sys

value = sys.argv[1]
max_images_per_request = int(sys.argv[2])
missed_creatures_max_images_per_request = int(sys.argv[3])
fill_missing_masks = sys.argv[4] == "1"
find_missed_creatures = sys.argv[5] == "1"
try:
    parsed = json.loads(value)
except Exception as exc:
    raise SystemExit(f"Invalid --limit-mm-per-prompt JSON: {value}\n{exc}")

if not isinstance(parsed, dict):
    raise SystemExit(
        f"Invalid --limit-mm-per-prompt JSON: expected object, got {type(parsed).__name__}"
    )

requested_image_budget = max(1, max_images_per_request)
if fill_missing_masks:
    requested_image_budget = max(requested_image_budget, 3)
if find_missed_creatures:
    requested_image_budget = max(
        requested_image_budget, missed_creatures_max_images_per_request
    )

current_image_limit = parsed.get("image", 0)
try:
    current_image_limit = int(current_image_limit)
except Exception:
    current_image_limit = 0
parsed["image"] = max(current_image_limit, requested_image_budget)

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

stages_csv=""
if [[ ${#stages[@]} -gt 0 ]]; then
  stages_csv="$(IFS=,; echo "${stages[*]}")"
fi

if [[ ! -f "$SBATCH_TEMPLATE" ]]; then
  echo "Missing sbatch template: $SBATCH_TEMPLATE" >&2
  exit 1
fi

if [[ -z "$output_path" ]]; then
  output_path="${SCRATCH:-/scratch/$USER}/sam3/logs/id_reassign/${batch_safe}/out/%x-%j.out"
fi
if [[ -z "$error_path" ]]; then
  error_path="${SCRATCH:-/scratch/$USER}/sam3/logs/id_reassign/${batch_safe}/err/%x-%j.err"
fi
if [[ "$dry_run" != "1" ]]; then
  mkdir -p "$(dirname "$output_path")" "$(dirname "$error_path")"
fi

sbatch_cmd=(sbatch --parsable)
[[ -n "$limit_mm_per_prompt_b64" ]] && sbatch_cmd+=(--export "ALL,SUBMITTED_LIMIT_MM_PER_PROMPT_B64=$limit_mm_per_prompt_b64,INPUT_DIRS_JSON_B64=$input_dirs_json_b64")
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
  "INPUT_DIRS_JSON=$input_dirs_json"
  "INPUT_DIRS_JSON_B64=$input_dirs_json_b64"
  "BATCH_SAFE=$batch_safe"
  "PROMPT_PROFILE=$prompt_profile"
  "PROMPT_PATH=$prompt_path"
  "MISSING_MASK_PROMPT_PATH=$missing_mask_prompt_path"
  "VERIFY_GAP_FILL_PROMPT_PATH=$verify_gap_fill_prompt_path"
  "MISSED_CREATURES_PROMPT_PATH=$missed_creatures_prompt_path"
  "VERIFY_MISSED_CREATURES_PROMPT_PATH=$verify_missed_creatures_prompt_path"
  "OUTLIER_MASK_PROMPT_PATH=$outlier_mask_prompt_path"
  "OUTPUT_SUBDIR=$output_subdir"
  "OUTPUT_VIDEO_NAME=$output_video_name"
  "MODEL_ID=$model_id"
  "MODEL_REVISION=$model_revision"
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
  "MAX_COMPLETION_TOKENS=$max_completion_tokens"
  "MAX_JSON_RETRIES=$max_json_retries"
  "WINDOW_SIZE=$window_size"
  "WINDOW_STRIDE=$window_stride"
  "ASSIGNMENT_HISTORY_FRAMES=$assignment_history_frames"
  "ASSIGNMENT_HEURISTIC_MIN_SCORE=$assignment_heuristic_min_score"
  "STAGES_CSV=$stages_csv"
  "FIND_MISSED_CREATURES=$find_missed_creatures"
  "MISSED_CREATURES_WINDOW_SIZE=$missed_creatures_window_size"
  "MISSED_CREATURES_WINDOW_STRIDE=$missed_creatures_window_stride"
  "MAX_MISSED_CREATURE_ISSUES_PER_WINDOW=$max_missed_creature_issues_per_window"
  "MISSED_CREATURES_MAX_ROUNDS=$missed_creatures_max_rounds"
  "MISSED_CREATURES_MAX_ATTEMPTS=$missed_creatures_max_attempts"
  "MISSED_CREATURES_MAX_IMAGES_PER_REQUEST=$missed_creatures_max_images_per_request"
  "MISSED_CREATURES_DUPLICATE_IOU_THRESHOLD=$missed_creatures_duplicate_iou_threshold"
  "MAX_GAP_ISSUES_PER_WINDOW=$max_gap_issues_per_window"
  "GAP_FILL_MAX_ATTEMPTS=$gap_fill_max_attempts"
  "GAP_FILL_POINT_CANDIDATES=$gap_fill_point_candidates"
  "MAX_OUTLIER_CHECKS_PER_WINDOW=$max_outlier_checks_per_window"
  "IMAGE_DETAIL=$image_detail"
  "MAX_IMAGES_PER_REQUEST=$max_images_per_request"
  "IMAGE_MAX_EDGE=$image_max_edge"
  "IMAGE_MIN_EDGE=$image_min_edge"
  "COLLAGE_COLS=$collage_cols"
  "COLLAGE_TILE_MAX_EDGE=$collage_tile_max_edge"
  "SAM3_GPU_IDS=$sam3_gpu_ids"
  "SAM3_IMAGE_SIZE=$sam3_image_size"
  "SAM3_OFFLOAD_VIDEO_TO_CPU=$sam3_offload_video_to_cpu"
  "FILL_MISSING_MASKS=$fill_missing_masks"
  "FILTER_OUTLIER_MASKS=$filter_outlier_masks"
  "RENDER_VIDEO=$render_video"
  "DEBUG=$debug"
  "CONTINUE_ON_ERROR=$continue_on_error"
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
echo "Batch log folder: ${SCRATCH:-/scratch/$USER}/sam3/logs/id_reassign/${batch_safe}"
