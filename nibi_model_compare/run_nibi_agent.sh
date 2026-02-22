#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SUBMIT_WRAPPER="${REPO_ROOT}/nibi_model_compare/slurm/submit_nibi_job.sh"
RUNNER_PY="${REPO_ROOT}/nibi_model_compare/run_video_agent_openai.py"

usage() {
  cat <<'EOF'
Run SAM3 agent workflow on Nibi in either:
  1) interactive mode (inside an existing Slurm allocation), or
  2) submit mode (submit an sbatch job non-interactively).

Usage:
  nibi_model_compare/run_nibi_agent.sh [options]

Core options:
  --mode <auto|interactive|submit>   Default: auto
  --account <name>                   Default: $ACCOUNT or rpp-kmoran
  --video-path <path>                Default: /project/$ACCOUNT/$USER/data/onc/chinacreekclipped.mp4
  --prompt <text>                    Default: identify and segment small creatures in the underwater scene
  --model-id <hf-model-id>           Default: Qwen/Qwen3-VL-30B-A3B-Instruct
  --output-dir <path>                Interactive mode output directory
  --output-root <path>               Submit mode root (passed to submit wrapper)
  --env-file <path>                  Default: <repo>/.env
  --venv-path <path>                 Default: <repo>/.venv
  --dry-run                          Print commands only
  -h, --help                         Show help

vLLM options:
  --port <n>                         Default: 8001
  --tp-size <n>                      Default: 1
  --max-model-len <n>                Default: 16384
  --max-num-seqs <n>                 Default: 1
  --gpu-memory-utilization <f>       Default: 0.92
  --limit-mm-per-prompt <json>       Default: {"image":1,"video":0}
  --vllm-cuda-visible-devices <ids>  Default: 0

Runner options:
  --runner-cuda-visible-devices <ids> Default: 1
  --runner-gpu-ids <ids>              Default: 0
  --image-size <n>                     Default: 1008
  --max-completion-tokens <n>          Default: 1024
  --save-prompts                       Save prompts only, skip propagation
  --debug                              Enable debug logs

SAM3 env toggles (forwarded in both modes):
  --sam3-disable-warmup <0|1>                  Default: 1
  --sam3-max-images-per-request <n>            Default: 1
  --pytorch-cuda-alloc-conf <value>            Default: expandable_segments:True
  --sam3-overlay-max-mask-area-ratio <float>   Optional
  --sam3-overlay-alpha <float>                 Optional

Submit-mode Slurm overrides:
  --time <hh:mm:ss>                   Default: 02:00:00
  --gpus-per-node <spec>              Default: h100:2
  --cpus-per-task <n>                 Default: 16
  --mem <spec>                        Default: 128000M
  --template <single|array|tp8>       Default: single

Examples:
  # In an existing interactive allocation:
  nibi_model_compare/run_nibi_agent.sh --mode interactive --debug

  # Submit a batch job:
  nibi_model_compare/run_nibi_agent.sh \
    --mode submit \
    --account rpp-kmoran \
    --video-path /project/rpp-kmoran/$USER/data/onc/chinacreekclipped.mp4 \
    --debug
EOF
}

mode="auto"
account="${ACCOUNT:-rpp-kmoran}"
video_path=""
prompt="identify and segment small creatures in the underwater scene"
model_id="Qwen/Qwen3-VL-30B-A3B-Instruct"
output_dir=""
output_root=""
env_file="${REPO_ROOT}/.env"
venv_path="${REPO_ROOT}/.venv"
dry_run=0

port="8001"
tp_size="1"
max_model_len="16384"
max_num_seqs="1"
gpu_memory_utilization="0.92"
limit_mm_per_prompt='{"image":1,"video":0}'
vllm_cuda_visible_devices="0"

runner_cuda_visible_devices="1"
runner_gpu_ids="0"
image_size="1008"
max_completion_tokens="1024"
save_prompts=0
debug=0

sam3_disable_warmup="1"
sam3_max_images_per_request="1"
pytorch_cuda_alloc_conf="expandable_segments:True"
sam3_overlay_max_mask_area_ratio=""
sam3_overlay_alpha=""

time_limit="02:00:00"
gpus_per_node="h100:2"
cpus_per_task="16"
mem="128000M"
template="single"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) mode="$2"; shift 2 ;;
    --account) account="$2"; shift 2 ;;
    --video-path) video_path="$2"; shift 2 ;;
    --prompt) prompt="$2"; shift 2 ;;
    --model-id) model_id="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    --env-file) env_file="$2"; shift 2 ;;
    --venv-path) venv_path="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;

    --port) port="$2"; shift 2 ;;
    --tp-size) tp_size="$2"; shift 2 ;;
    --max-model-len) max_model_len="$2"; shift 2 ;;
    --max-num-seqs) max_num_seqs="$2"; shift 2 ;;
    --gpu-memory-utilization) gpu_memory_utilization="$2"; shift 2 ;;
    --limit-mm-per-prompt) limit_mm_per_prompt="$2"; shift 2 ;;
    --vllm-cuda-visible-devices) vllm_cuda_visible_devices="$2"; shift 2 ;;

    --runner-cuda-visible-devices) runner_cuda_visible_devices="$2"; shift 2 ;;
    --runner-gpu-ids) runner_gpu_ids="$2"; shift 2 ;;
    --image-size) image_size="$2"; shift 2 ;;
    --max-completion-tokens) max_completion_tokens="$2"; shift 2 ;;
    --save-prompts) save_prompts=1; shift ;;
    --debug) debug=1; shift ;;

    --sam3-disable-warmup) sam3_disable_warmup="$2"; shift 2 ;;
    --sam3-max-images-per-request) sam3_max_images_per_request="$2"; shift 2 ;;
    --pytorch-cuda-alloc-conf) pytorch_cuda_alloc_conf="$2"; shift 2 ;;
    --sam3-overlay-max-mask-area-ratio) sam3_overlay_max_mask_area_ratio="$2"; shift 2 ;;
    --sam3-overlay-alpha) sam3_overlay_alpha="$2"; shift 2 ;;

    --time) time_limit="$2"; shift 2 ;;
    --gpus-per-node) gpus_per_node="$2"; shift 2 ;;
    --cpus-per-task) cpus_per_task="$2"; shift 2 ;;
    --mem) mem="$2"; shift 2 ;;
    --template) template="$2"; shift 2 ;;

    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$video_path" ]]; then
  video_path="/project/${account}/${USER}/data/onc/chinacreekclipped.mp4"
fi

if [[ -z "$output_root" ]]; then
  output_root="/project/${account}/${USER}/sam3/runs"
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
if [[ -z "$output_dir" ]]; then
  output_dir="${output_root}/interactive_${SLURM_JOB_ID:-manual}_${timestamp}"
fi

if [[ "$mode" == "auto" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    mode="interactive"
  else
    mode="submit"
  fi
fi

run_interactive() {
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Interactive mode requires an active Slurm allocation (SLURM_JOB_ID not set)."
    echo "Use --mode submit from a login node, or obtain an allocation with salloc first."
    exit 1
  fi

  cd "$REPO_ROOT"
  if [[ ! -d "$venv_path" ]]; then
    echo "Virtual environment not found: $venv_path"
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${venv_path}/bin/activate"

  if [[ -f "$env_file" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
  fi

  export ACCOUNT="$account"
  export MODEL_ID="$model_id"
  export PORT="$port"
  export VIDEO_PATH="$video_path"
  export OUT_DIR="$output_dir"

  export JOB_CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/sam3-cache"
  export HF_HOME="${JOB_CACHE_ROOT}/hf"
  export HF_HUB_CACHE="${HF_HOME}/hub"
  export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
  export HF_XET_CACHE="${HF_HOME}/xet"
  export TORCH_HOME="${HF_HOME}/torch"
  export XDG_CACHE_HOME="${HF_HOME}/xdg"
  export TMPDIR="${SLURM_TMPDIR:-${JOB_CACHE_ROOT}/tmp}"
  export HF_HUB_DISABLE_XET=1
  unset TRANSFORMERS_CACHE
  mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TMPDIR"

  export SAM3_DISABLE_WARMUP="$sam3_disable_warmup"
  export SAM3_MAX_IMAGES_PER_REQUEST="$sam3_max_images_per_request"
  export PYTORCH_CUDA_ALLOC_CONF="$pytorch_cuda_alloc_conf"
  if [[ -n "$sam3_overlay_max_mask_area_ratio" ]]; then
    export SAM3_OVERLAY_MAX_MASK_AREA_RATIO="$sam3_overlay_max_mask_area_ratio"
  fi
  if [[ -n "$sam3_overlay_alpha" ]]; then
    export SAM3_OVERLAY_ALPHA="$sam3_overlay_alpha"
  fi

  mkdir -p "$OUT_DIR"
  pkill -f "vllm serve" || true

  vllm_log="/tmp/vllm_${SLURM_JOB_ID}.log"
  runner_log="/tmp/sam3_runner_${SLURM_JOB_ID}.log"

  vllm_cmd=(
    vllm serve "$MODEL_ID"
    --tensor-parallel-size "$tp_size"
    --allowed-local-media-path /
    --gpu-memory-utilization "$gpu_memory_utilization"
    --max-model-len "$max_model_len"
    --max-num-seqs "$max_num_seqs"
    --limit-mm-per-prompt "$limit_mm_per_prompt"
    --port "$PORT"
  )

  runner_cmd=(
    python "$RUNNER_PY"
    --video_path "$VIDEO_PATH"
    --prompt "$prompt"
    --server_url "http://127.0.0.1:${PORT}/v1"
    --model "$MODEL_ID"
    --output_dir "$OUT_DIR"
    --gpus "$runner_gpu_ids"
    --image_size "$image_size"
    --max_completion_tokens "$max_completion_tokens"
  )
  if [[ "$save_prompts" == "1" ]]; then
    runner_cmd+=(--save_prompts)
  fi
  if [[ "$debug" == "1" ]]; then
    runner_cmd+=(--debug)
  fi

  echo "=== Interactive run configuration ==="
  echo "video: $VIDEO_PATH"
  echo "model: $MODEL_ID"
  echo "out:   $OUT_DIR"
  echo "vllm cuda visible: $vllm_cuda_visible_devices"
  echo "runner cuda visible: $runner_cuda_visible_devices"

  if [[ "$dry_run" == "1" ]]; then
    echo "CUDA_VISIBLE_DEVICES=${vllm_cuda_visible_devices} ${vllm_cmd[*]} >${vllm_log} 2>&1 &"
    echo "CUDA_VISIBLE_DEVICES=${runner_cuda_visible_devices} ${runner_cmd[*]} 2>&1 | tee ${runner_log}"
    exit 0
  fi

  CUDA_VISIBLE_DEVICES="$vllm_cuda_visible_devices" "${vllm_cmd[@]}" >"$vllm_log" 2>&1 &
  vllm_pid=$!

  cleanup() {
    kill "$vllm_pid" 2>/dev/null || true
  }
  trap cleanup EXIT

  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
      break
    fi
    if ! kill -0 "$vllm_pid" 2>/dev/null; then
      echo "vLLM failed to start. Last log lines:"
      tail -n 120 "$vllm_log" || true
      exit 1
    fi
    sleep 2
  done
  curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null

  CUDA_VISIBLE_DEVICES="$runner_cuda_visible_devices" "${runner_cmd[@]}" 2>&1 | tee "$runner_log"

  if [[ -f "$OUT_DIR/run_metrics.json" ]]; then
    cat "$OUT_DIR/run_metrics.json"
  fi
  echo "Done. Output directory: $OUT_DIR"
}

run_submit() {
  if [[ ! -x "$SUBMIT_WRAPPER" ]]; then
    echo "Submit wrapper not found or not executable: $SUBMIT_WRAPPER"
    exit 1
  fi

  submit_cmd=(
    "$SUBMIT_WRAPPER"
    --template "$template"
    --account "$account"
    --time "$time_limit"
    --gpus-per-node "$gpus_per_node"
    --cpus-per-task "$cpus_per_task"
    --mem "$mem"
    --video-path "$video_path"
    --prompt "$prompt"
    --model-id "$model_id"
    --server-port "$port"
    --tp-size "$tp_size"
    --vllm-cuda-visible-devices "$vllm_cuda_visible_devices"
    --runner-cuda-visible-devices "$runner_cuda_visible_devices"
    --runner-gpu-ids "$runner_gpu_ids"
    --gpu-memory-utilization "$gpu_memory_utilization"
    --max-model-len "$max_model_len"
    --max-num-seqs "$max_num_seqs"
    --limit-mm-per-prompt "$limit_mm_per_prompt"
    --image-size "$image_size"
    --max-completion-tokens "$max_completion_tokens"
    --env-file "$env_file"
    --venv-path "$venv_path"
    --repo-root "$REPO_ROOT"
    --project-root "$REPO_ROOT"
    --out-root "$output_root"
    --set-env "SAM3_DISABLE_WARMUP=${sam3_disable_warmup}"
    --set-env "SAM3_MAX_IMAGES_PER_REQUEST=${sam3_max_images_per_request}"
    --set-env "PYTORCH_CUDA_ALLOC_CONF=${pytorch_cuda_alloc_conf}"
  )
  if [[ -n "$sam3_overlay_max_mask_area_ratio" ]]; then
    submit_cmd+=(--set-env "SAM3_OVERLAY_MAX_MASK_AREA_RATIO=${sam3_overlay_max_mask_area_ratio}")
  fi
  if [[ -n "$sam3_overlay_alpha" ]]; then
    submit_cmd+=(--set-env "SAM3_OVERLAY_ALPHA=${sam3_overlay_alpha}")
  fi
  if [[ "$save_prompts" == "1" ]]; then
    submit_cmd+=(--save-prompts)
  fi
  if [[ "$debug" == "1" ]]; then
    submit_cmd+=(--debug)
  fi
  if [[ "$dry_run" == "1" ]]; then
    submit_cmd+=(--dry-run)
  fi

  echo "=== Submit command ==="
  printf '%q ' "${submit_cmd[@]}"
  echo
  "${submit_cmd[@]}"
}

case "$mode" in
  interactive) run_interactive ;;
  submit) run_submit ;;
  *)
    echo "Invalid mode: $mode"
    usage
    exit 1
    ;;
esac

