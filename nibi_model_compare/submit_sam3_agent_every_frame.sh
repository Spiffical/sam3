#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SBATCH_TEMPLATE="${REPO_ROOT}/nibi_model_compare/slurm/sam3_agent_every_frame.sbatch"

usage() {
  cat <<'EOF'
Submit a Slurm job that runs the SAM3 agent loop on every frame of a video.

Usage:
  nibi_model_compare/submit_sam3_agent_every_frame.sh [options] --video-path /path/to/video.mp4

Required:
  --video-path <path>

Common options:
  --account <name>                    Default: $ACCOUNT or rpp-kmoran
  --video-path <path>                 Input video
  --prompt <text>                     Default: small creatures
  --prompt-profile <name>             Default: underwater
  --model-id <hf-model-id>            Default: Qwen/Qwen3.5-27B
  --model-revision <rev>              Optional HF revision
  --venv-path <path>                  Default: <repo>/.venv
  --env-file <path>                   Default: <repo>/.env
  --output-root <path>                Default: /scratch/$USER/sam3/runs
  --job-name <name>                   Default: sam3_agent_frames
  --dry-run                           Print env + sbatch command only
  -h, --help                          Show help

LLM/vLLM options:
  --server-port <n>                   Default: 8006
  --tp-size <n>                       Default: 1
  --max-model-len <n>                 Default: 16384
  --max-num-seqs <n>                  Default: 1
  --gpu-memory-utilization <f>        Default: 0.90
  --limit-mm-per-prompt <json>        Default: {"image":1,"video":0}
  --vllm-runtime <auto|venv|apptainer> Default: auto
  --apptainer-image <path>            Optional SIF path
  --vllm-cuda-visible-devices <ids>   Default: 0

Runner options:
  --runner-cuda-visible-devices <ids> Default: 1
  --runner-gpu-ids <ids>              Default: 0
  --device <cuda|cpu>                 Default: cuda
  --max-generations <n>               Default: 10
  --max-completion-tokens <n>         Default: 1024
  --max-frames <n>                    Optional frame cap
  --checkpoint-path <path>            Optional local SAM3 checkpoint
  --system-prompt-path <path>         Optional override for base system prompt
  --iterative-system-prompt-path <path> Optional override for iterative system prompt
  --debug                             Keep full per-frame agent debug artifacts
  --keep-artifacts                    Preserve per-frame images and agent folders
  --continue-on-error                 Continue on frame-level agent errors (default)
  --no-continue-on-error              Fail the job on the first frame-level error

Slurm options:
  --partition <name>                  Optional partition
  --gpus-per-node <spec>              Default: h100:2
  --cpus-per-task <n>                 Default: 16
  --mem <spec>                        Default: 128000M
  --time <hh:mm:ss>                   Default: 08:00:00
  --output <path>                     Default: /scratch/$USER/sam3/logs/agent_every_frame/out/%x-%j.out
  --error <path>                      Default: /scratch/$USER/sam3/logs/agent_every_frame/err/%x-%j.err

Example:
  nibi_model_compare/submit_sam3_agent_every_frame.sh \
    --account rpp-kmoran \
    --venv-path "$HOME/sam3/.venv-qwen35" \
    --vllm-runtime apptainer \
    --apptainer-image "${SCRATCH:-/scratch/$USER}/vllm-openai-nightly.sif" \
    --gpus-per-node h100:2 \
    --cpus-per-task 16 \
    --mem 128000M \
    --time 08:00:00 \
    --model-id "Qwen/Qwen3.5-27B" \
    --server-port 8006 \
    --tp-size 1 \
    --vllm-cuda-visible-devices 0 \
    --runner-cuda-visible-devices 1 \
    --runner-gpu-ids 0 \
    --max-model-len 16384 \
    --max-num-seqs 1 \
    --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image":1,"video":0}' \
    --video-path /project/rpp-kmoran/$USER/data/onc/input.mp4 \
    --prompt "small creatures" \
    --prompt-profile underwater \
    --max-generations 10 \
    --max-completion-tokens 1024 \
    --continue-on-error
EOF
}

account="${ACCOUNT:-rpp-kmoran}"
partition=""
video_path=""
prompt="small creatures"
prompt_profile="underwater"
model_id="Qwen/Qwen3.5-27B"
model_revision=""
venv_path="${REPO_ROOT}/.venv"
env_file="${REPO_ROOT}/.env"
output_root="${SCRATCH:-/scratch/$USER}/sam3/runs"
job_name="sam3_agent_frames"
dry_run=0

server_port="8006"
tp_size="1"
max_model_len="16384"
max_num_seqs="1"
gpu_memory_utilization="0.90"
limit_mm_per_prompt='{"image":1,"video":0}'
vllm_runtime="auto"
apptainer_image=""
vllm_cuda_visible_devices="0"

runner_cuda_visible_devices="1"
runner_gpu_ids="0"
device="cuda"
max_generations="10"
max_completion_tokens="1024"
max_frames=""
checkpoint_path=""
system_prompt_path=""
iterative_system_prompt_path=""
debug=0
keep_artifacts=0
continue_on_error=1

gpus_per_node="h100:2"
cpus_per_task="16"
mem="128000M"
time_limit="08:00:00"
output_path=""
error_path=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --account) account="$2"; shift 2 ;;
    --partition) partition="$2"; shift 2 ;;
    --video-path) video_path="$2"; shift 2 ;;
    --prompt) prompt="$2"; shift 2 ;;
    --prompt-profile) prompt_profile="$2"; shift 2 ;;
    --model-id) model_id="$2"; shift 2 ;;
    --model-revision) model_revision="$2"; shift 2 ;;
    --venv-path) venv_path="$2"; shift 2 ;;
    --env-file) env_file="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    --job-name) job_name="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;

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
    --runner-gpu-ids) runner_gpu_ids="$2"; shift 2 ;;
    --device) device="$2"; shift 2 ;;
    --max-generations) max_generations="$2"; shift 2 ;;
    --max-completion-tokens) max_completion_tokens="$2"; shift 2 ;;
    --max-frames) max_frames="$2"; shift 2 ;;
    --checkpoint-path) checkpoint_path="$2"; shift 2 ;;
    --system-prompt-path) system_prompt_path="$2"; shift 2 ;;
    --iterative-system-prompt-path) iterative_system_prompt_path="$2"; shift 2 ;;
    --debug) debug=1; shift ;;
    --keep-artifacts) keep_artifacts=1; shift ;;
    --continue-on-error) continue_on_error=1; shift ;;
    --no-continue-on-error) continue_on_error=0; shift ;;

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

if [[ -z "$video_path" ]]; then
  echo "--video-path is required." >&2
  usage
  exit 1
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

if [[ ! -f "$SBATCH_TEMPLATE" ]]; then
  echo "Missing sbatch template: $SBATCH_TEMPLATE" >&2
  exit 1
fi

if [[ -z "$output_path" ]]; then
  output_path="${SCRATCH:-/scratch/$USER}/sam3/logs/agent_every_frame/out/%x-%j.out"
fi
if [[ -z "$error_path" ]]; then
  error_path="${SCRATCH:-/scratch/$USER}/sam3/logs/agent_every_frame/err/%x-%j.err"
fi
if [[ "$dry_run" != "1" ]]; then
  mkdir -p "$(dirname "$output_path")" "$(dirname "$error_path")"
fi

sbatch_cmd=(sbatch --parsable)
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
  "VIDEO_PATH=$video_path"
  "PROMPT=$prompt"
  "PROMPT_PROFILE=$prompt_profile"
  "MODEL_ID=$model_id"
  "MODEL_REVISION=$model_revision"
  "OUT_ROOT=$output_root"
  "SERVER_PORT=$server_port"
  "TP_SIZE=$tp_size"
  "MAX_MODEL_LEN=$max_model_len"
  "MAX_NUM_SEQS=$max_num_seqs"
  "GPU_MEMORY_UTILIZATION=$gpu_memory_utilization"
  "SUBMITTED_LIMIT_MM_PER_PROMPT=$limit_mm_per_prompt"
  "VLLM_RUNTIME=$vllm_runtime"
  "APPTAINER_IMAGE=$apptainer_image"
  "VLLM_CUDA_VISIBLE_DEVICES=$vllm_cuda_visible_devices"
  "RUNNER_CUDA_VISIBLE_DEVICES=$runner_cuda_visible_devices"
  "RUNNER_GPU_IDS=$runner_gpu_ids"
  "DEVICE=$device"
  "MAX_GENERATIONS=$max_generations"
  "MAX_COMPLETION_TOKENS=$max_completion_tokens"
  "DEBUG=$debug"
  "KEEP_ARTIFACTS=$keep_artifacts"
  "CONTINUE_ON_ERROR=$continue_on_error"
)
[[ -n "$max_frames" ]] && env_vars+=("MAX_FRAMES=$max_frames")
[[ -n "$checkpoint_path" ]] && env_vars+=("CHECKPOINT_PATH=$checkpoint_path")
[[ -n "$system_prompt_path" ]] && env_vars+=("SYSTEM_PROMPT_PATH=$system_prompt_path")
[[ -n "$iterative_system_prompt_path" ]] && env_vars+=("ITERATIVE_SYSTEM_PROMPT_PATH=$iterative_system_prompt_path")

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
