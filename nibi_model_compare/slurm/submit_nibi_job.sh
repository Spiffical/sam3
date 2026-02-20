#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  nibi_model_compare/slurm/submit_nibi_job.sh [options]

Core options:
  --template {single|array|tp8}    Which sbatch template to submit (default: single)
  --dry-run                         Print env + sbatch command only
  -h, --help                        Show this help

Slurm resource overrides:
  --account <name>
  --partition <name>
  --nodes <n>
  --gpus-per-node <spec>           e.g. h100:4
  --cpus-per-task <n>
  --mem <spec>                     e.g. 128000M
  --time <hh:mm:ss>
  --array <spec>                   e.g. 0-9 (array template)
  --job-name <name>
  --output <path>
  --error <path>
  --sbatch-opt <raw-opt>           Repeatable extra sbatch option

Runtime overrides (forwarded as env vars):
  --repo-root <path>
  --project-root <path>
  --default-project-prefix <path>
  --project-cache-root <path>
  --persistent-cache-root <path>
  --job-cache-root <path>
  --log-root <path>
  --env-file <path>
  --venv-path <path>
  --video-path <path>
  --videos-manifest <path>
  --prompt <text>
  --model-id <hf-model-id>
  --server-port <port>
  --tp-size <n>
  --gpu-ids <ids>                  e.g. 0,1,2,3 (passed to run_video_agent_openai.py)
  --runner-gpu-ids <ids>
  --vllm-cuda-visible-devices <ids>
  --runner-cuda-visible-devices <ids>
  --gpu-memory-utilization <float> e.g. 0.90
  --max-model-len <n>
  --max-num-seqs <n>
  --limit-mm-per-prompt <json>     e.g. '{"image":1,"video":0}'
  --image-size <n>
  --max-completion-tokens <n>
  --save-prompts / --no-save-prompts
  --debug / --no-debug
  --out-root <path>

Useful pass-through env knobs:
  --sam3-max-images-per-request <n>
  --sam3-image-detail <low|high|auto>
  --sam3-agent-image-max-edge <n>
  --pytorch-cuda-alloc-conf <value>
  --hf-hub-disable-xet <0|1>
  --set-env <KEY=VALUE>            Repeatable extra env var forwarded to sbatch job

Examples:
  # Single-model run with defaults from template profile
  nibi_model_compare/slurm/submit_nibi_job.sh \
    --template single \
    --account <account> \
    --video-path /project/<account>/$USER/data/onc/chinacreekclipped.mp4

  # Two-GPU split: vLLM on GPU0, SAM3 on GPU1
  nibi_model_compare/slurm/submit_nibi_job.sh \
    --template single \
    --account <account> \
    --gpus-per-node h100:2 \
    --tp-size 1 \
    --vllm-cuda-visible-devices 0 \
    --runner-cuda-visible-devices 1 \
    --runner-gpu-ids 0 \
    --limit-mm-per-prompt '{"image":1,"video":0}' \
    --max-completion-tokens 512 \
    --debug

  # Array run with a custom manifest
  nibi_model_compare/slurm/submit_nibi_job.sh \
    --template array \
    --account <account> \
    --array 0-31 \
    --videos-manifest nibi_model_compare/videos_manifest.txt
EOF
}

template="single"
dry_run=0

# Slurm options
account=""
partition=""
nodes=""
gpus_per_node=""
cpus_per_task=""
mem=""
time_limit=""
array_spec=""
job_name=""
output_path=""
error_path=""
sbatch_extra_opts=()

# Runtime options
default_repo_root="$(cd "${SCRIPT_DIR}/../.." && pwd)"
repo_root="${default_repo_root}"
project_root="$repo_root"
default_project_prefix=""
project_cache_root=""
persistent_cache_root=""
job_cache_root=""
log_root=""
env_file="${repo_root}/.env"
venv_path="${repo_root}/.venv"
video_path="${repo_root}/assets/videos/chinacreekclipped.mp4"
videos_manifest="nibi_model_compare/videos_manifest.txt"
prompt="segment all visible marine organisms"
model_id=""
server_port=""
tp_size=""
gpu_ids=""
runner_gpu_ids=""
vllm_cuda_visible_devices=""
runner_cuda_visible_devices=""
gpu_memory_utilization=""
max_model_len=""
max_num_seqs=""
limit_mm_per_prompt=""
image_size="1024"
max_completion_tokens="1024"
save_prompts="0"
debug="0"
out_root=""

hf_hub_disable_xet="1"
sam3_max_images_per_request=""
sam3_image_detail=""
sam3_agent_image_max_edge=""
pytorch_cuda_alloc_conf=""
extra_env_vars=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --template) template="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;

    --account) account="$2"; shift 2 ;;
    --partition) partition="$2"; shift 2 ;;
    --nodes) nodes="$2"; shift 2 ;;
    --gpus-per-node) gpus_per_node="$2"; shift 2 ;;
    --cpus-per-task) cpus_per_task="$2"; shift 2 ;;
    --mem) mem="$2"; shift 2 ;;
    --time) time_limit="$2"; shift 2 ;;
    --array) array_spec="$2"; shift 2 ;;
    --job-name) job_name="$2"; shift 2 ;;
    --output) output_path="$2"; shift 2 ;;
    --error) error_path="$2"; shift 2 ;;
    --sbatch-opt) sbatch_extra_opts+=("$2"); shift 2 ;;

    --repo-root) repo_root="$2"; shift 2 ;;
    --project-root) project_root="$2"; shift 2 ;;
    --default-project-prefix) default_project_prefix="$2"; shift 2 ;;
    --project-cache-root) project_cache_root="$2"; shift 2 ;;
    --persistent-cache-root) persistent_cache_root="$2"; shift 2 ;;
    --job-cache-root) job_cache_root="$2"; shift 2 ;;
    --log-root) log_root="$2"; shift 2 ;;
    --env-file) env_file="$2"; shift 2 ;;
    --venv-path) venv_path="$2"; shift 2 ;;
    --video-path) video_path="$2"; shift 2 ;;
    --videos-manifest) videos_manifest="$2"; shift 2 ;;
    --prompt) prompt="$2"; shift 2 ;;
    --model-id) model_id="$2"; shift 2 ;;
    --server-port) server_port="$2"; shift 2 ;;
    --tp-size) tp_size="$2"; shift 2 ;;
    --gpu-ids) gpu_ids="$2"; shift 2 ;;
    --runner-gpu-ids) runner_gpu_ids="$2"; shift 2 ;;
    --vllm-cuda-visible-devices) vllm_cuda_visible_devices="$2"; shift 2 ;;
    --runner-cuda-visible-devices) runner_cuda_visible_devices="$2"; shift 2 ;;
    --gpu-memory-utilization) gpu_memory_utilization="$2"; shift 2 ;;
    --max-model-len) max_model_len="$2"; shift 2 ;;
    --max-num-seqs) max_num_seqs="$2"; shift 2 ;;
    --limit-mm-per-prompt) limit_mm_per_prompt="$2"; shift 2 ;;
    --image-size) image_size="$2"; shift 2 ;;
    --max-completion-tokens) max_completion_tokens="$2"; shift 2 ;;
    --save-prompts) save_prompts="1"; shift ;;
    --no-save-prompts) save_prompts="0"; shift ;;
    --debug) debug="1"; shift ;;
    --no-debug) debug="0"; shift ;;
    --out-root) out_root="$2"; shift 2 ;;

    --sam3-max-images-per-request) sam3_max_images_per_request="$2"; shift 2 ;;
    --sam3-image-detail) sam3_image_detail="$2"; shift 2 ;;
    --sam3-agent-image-max-edge) sam3_agent_image_max_edge="$2"; shift 2 ;;
    --pytorch-cuda-alloc-conf) pytorch_cuda_alloc_conf="$2"; shift 2 ;;
    --hf-hub-disable-xet) hf_hub_disable_xet="$2"; shift 2 ;;
    --set-env)
      if [[ "$2" != *=* ]]; then
        echo "--set-env expects KEY=VALUE, got: $2"
        exit 1
      fi
      extra_env_vars+=("$2")
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

template_script=""
case "$template" in
  single)
    template_script="${SCRIPT_DIR}/nibi_single_model.sbatch"
    : "${model_id:=Qwen/Qwen3-VL-30B-A3B-Instruct}"
    : "${server_port:=8001}"
    : "${tp_size:=4}"
    : "${gpu_memory_utilization:=0.90}"
    : "${job_name:=sam3_nibi_single}"
    : "${nodes:=1}"
    : "${gpus_per_node:=h100:4}"
    : "${cpus_per_task:=16}"
    : "${mem:=128000M}"
    : "${time_limit:=08:00:00}"
    ;;
  array)
    template_script="${SCRIPT_DIR}/nibi_matrix_array.sbatch"
    : "${model_id:=Qwen/Qwen3-VL-30B-A3B-Instruct}"
    : "${server_port:=8001}"
    : "${tp_size:=4}"
    : "${gpu_memory_utilization:=0.90}"
    : "${job_name:=sam3_nibi_array}"
    : "${nodes:=1}"
    : "${gpus_per_node:=h100:4}"
    : "${cpus_per_task:=16}"
    : "${mem:=128000M}"
    : "${time_limit:=10:00:00}"
    : "${array_spec:=0-9}"
    ;;
  tp8)
    template_script="${SCRIPT_DIR}/nibi_qwen35_tp8_candidate.sbatch"
    : "${model_id:=Qwen/Qwen3.5-397B-A17B}"
    : "${server_port:=8005}"
    : "${tp_size:=8}"
    : "${gpu_memory_utilization:=0.8}"
    : "${max_model_len:=32768}"
    if [[ -z "${limit_mm_per_prompt}" ]]; then
      limit_mm_per_prompt='{"image":4,"video":2}'
    fi
    : "${job_name:=sam3_qwen35_tp8}"
    : "${nodes:=1}"
    : "${gpus_per_node:=h100:8}"
    : "${cpus_per_task:=32}"
    : "${mem:=256000M}"
    : "${time_limit:=10:00:00}"
    ;;
  *)
    echo "Unsupported template: $template"
    usage
    exit 1
    ;;
esac

if [[ ! -f "$template_script" ]]; then
  echo "Template script not found: $template_script"
  exit 1
fi

sbatch_cmd=(sbatch --parsable)
[[ -n "$account" ]] && sbatch_cmd+=(--account "$account")
[[ -n "$partition" ]] && sbatch_cmd+=(--partition "$partition")
[[ -n "$nodes" ]] && sbatch_cmd+=(--nodes "$nodes")
[[ -n "$gpus_per_node" ]] && sbatch_cmd+=(--gpus-per-node "$gpus_per_node")
[[ -n "$cpus_per_task" ]] && sbatch_cmd+=(--cpus-per-task "$cpus_per_task")
[[ -n "$mem" ]] && sbatch_cmd+=(--mem "$mem")
[[ -n "$time_limit" ]] && sbatch_cmd+=(--time "$time_limit")
[[ -n "$array_spec" && "$template" == "array" ]] && sbatch_cmd+=(--array "$array_spec")
[[ -n "$job_name" ]] && sbatch_cmd+=(--job-name "$job_name")
[[ -n "$output_path" ]] && sbatch_cmd+=(--output "$output_path")
[[ -n "$error_path" ]] && sbatch_cmd+=(--error "$error_path")
for opt in "${sbatch_extra_opts[@]}"; do
  sbatch_cmd+=("$opt")
done
sbatch_cmd+=("$template_script")

env_vars=(
  "REPO_ROOT=$repo_root"
  "PROJECT_ROOT=$project_root"
  "ENV_FILE=$env_file"
  "VENV_PATH=$venv_path"
  "VIDEO_PATH=$video_path"
  "VIDEOS_MANIFEST=$videos_manifest"
  "PROMPT=$prompt"
  "MODEL_ID=$model_id"
  "SERVER_PORT=$server_port"
  "TP_SIZE=$tp_size"
  "IMAGE_SIZE=$image_size"
  "MAX_COMPLETION_TOKENS=$max_completion_tokens"
  "SAVE_PROMPTS=$save_prompts"
  "DEBUG=$debug"
  "HF_HUB_DISABLE_XET=$hf_hub_disable_xet"
)
[[ -n "$project_cache_root" ]] && env_vars+=("PROJECT_CACHE_ROOT=$project_cache_root")
[[ -n "$default_project_prefix" ]] && env_vars+=("DEFAULT_PROJECT_PREFIX=$default_project_prefix")
[[ -n "$persistent_cache_root" ]] && env_vars+=("PERSISTENT_CACHE_ROOT=$persistent_cache_root")
[[ -n "$job_cache_root" ]] && env_vars+=("JOB_CACHE_ROOT=$job_cache_root")
[[ -n "$log_root" ]] && env_vars+=("LOG_ROOT=$log_root")
[[ -n "$out_root" ]] && env_vars+=("OUT_ROOT=$out_root")
[[ -n "$gpu_ids" ]] && env_vars+=("GPU_IDS=$gpu_ids")
[[ -n "$runner_gpu_ids" ]] && env_vars+=("RUNNER_GPU_IDS=$runner_gpu_ids")
[[ -n "$vllm_cuda_visible_devices" ]] && env_vars+=("VLLM_CUDA_VISIBLE_DEVICES=$vllm_cuda_visible_devices")
[[ -n "$runner_cuda_visible_devices" ]] && env_vars+=("RUNNER_CUDA_VISIBLE_DEVICES=$runner_cuda_visible_devices")
[[ -n "$gpu_memory_utilization" ]] && env_vars+=("GPU_MEMORY_UTILIZATION=$gpu_memory_utilization")
[[ -n "$max_model_len" ]] && env_vars+=("MAX_MODEL_LEN=$max_model_len")
[[ -n "$max_num_seqs" ]] && env_vars+=("MAX_NUM_SEQS=$max_num_seqs")
[[ -n "$limit_mm_per_prompt" ]] && env_vars+=("LIMIT_MM_PER_PROMPT=$limit_mm_per_prompt")
[[ -n "$sam3_max_images_per_request" ]] && env_vars+=("SAM3_MAX_IMAGES_PER_REQUEST=$sam3_max_images_per_request")
[[ -n "$sam3_image_detail" ]] && env_vars+=("SAM3_IMAGE_DETAIL=$sam3_image_detail")
[[ -n "$sam3_agent_image_max_edge" ]] && env_vars+=("SAM3_AGENT_IMAGE_MAX_EDGE=$sam3_agent_image_max_edge")
[[ -n "$pytorch_cuda_alloc_conf" ]] && env_vars+=("PYTORCH_CUDA_ALLOC_CONF=$pytorch_cuda_alloc_conf")
env_vars+=("${extra_env_vars[@]}")

echo "Template: $template"
echo "Script:   $template_script"

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
