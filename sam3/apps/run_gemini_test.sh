#!/bin/bash
set -euo pipefail

# Ensure we are in the root directory if running from there, or handle paths
# Script location
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$(dirname "$DIR")")"
cd "$ROOT_DIR"

usage() {
    cat <<'EOF'
Usage: bash sam3/apps/run_gemini_test.sh [options] [api_key] [-- extra_args_for_python]

Options (CLI takes precedence over env):
  --video_path PATH
  --prompt TEXT
  --output_root DIR          Root folder for per-run outputs
  --output_dir DIR           Alias of --output_root
  --run_name NAME            Optional run folder name (default: run_YYYYmmdd_HHMMSS)
  --model NAME
  --analysis_second SEC      Which second to sample for agent analysis
  --gemini_api_mode MODE     auto|generate|chat
  --gemini_sdk MODE          auto|new|legacy
  --gemini_use_system_instruction 0|1
  --gemini_timeout_sec SEC
  --gemini_debug_dump | --no-gemini_debug_dump
  --gemini_debug_dir DIR
  --gemini_text_before_image 0|1
  --save_prompts
  --api_key KEY
  -h, --help

Env fallbacks:
  VIDEO_PATH, PROMPT, OUTPUT_DIR, GEMINI_MODEL, ANALYSIS_SECOND, GEMINI_API_MODE,
  SAM3_GEMINI_SDK,
  GEMINI_USE_SYSTEM_INSTRUCTION, GEMINI_TIMEOUT_SEC, GEMINI_DEBUG_DUMP,
  GEMINI_DEBUG_DIR, GEMINI_TEXT_BEFORE_IMAGE, SAVE_PROMPTS
EOF
}

# CLI candidates (optional; resolved after parse)
VIDEO_PATH_CLI=""
PROMPT_CLI=""
OUTPUT_ROOT_CLI=""
RUN_NAME_CLI=""
MODEL_CLI=""
ANALYSIS_SECOND_CLI=""
GEMINI_API_MODE_CLI=""
GEMINI_SDK_CLI=""
GEMINI_USE_SYSTEM_INSTRUCTION_CLI=""
GEMINI_TIMEOUT_SEC_CLI=""
GEMINI_DEBUG_DUMP_CLI=""
GEMINI_DEBUG_DIR_CLI=""
GEMINI_TEXT_BEFORE_IMAGE_CLI=""
SAVE_PROMPTS_CLI=""
API_KEY_CLI=""
EXTRA_ARGS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --video_path)
            VIDEO_PATH_CLI="$2"
            shift 2
            ;;
        --prompt)
            PROMPT_CLI="$2"
            shift 2
            ;;
        --output_root|--output_dir)
            OUTPUT_ROOT_CLI="$2"
            shift 2
            ;;
        --run_name)
            RUN_NAME_CLI="$2"
            shift 2
            ;;
        --model)
            MODEL_CLI="$2"
            shift 2
            ;;
        --analysis_second)
            ANALYSIS_SECOND_CLI="$2"
            shift 2
            ;;
        --gemini_api_mode)
            GEMINI_API_MODE_CLI="$2"
            shift 2
            ;;
        --gemini_sdk)
            GEMINI_SDK_CLI="$2"
            shift 2
            ;;
        --gemini_use_system_instruction)
            GEMINI_USE_SYSTEM_INSTRUCTION_CLI="$2"
            shift 2
            ;;
        --gemini_timeout_sec)
            GEMINI_TIMEOUT_SEC_CLI="$2"
            shift 2
            ;;
        --gemini_debug_dump)
            GEMINI_DEBUG_DUMP_CLI="1"
            shift
            ;;
        --no-gemini_debug_dump)
            GEMINI_DEBUG_DUMP_CLI="0"
            shift
            ;;
        --gemini_debug_dir)
            GEMINI_DEBUG_DIR_CLI="$2"
            shift 2
            ;;
        --gemini_text_before_image)
            GEMINI_TEXT_BEFORE_IMAGE_CLI="$2"
            shift 2
            ;;
        --save_prompts)
            SAVE_PROMPTS_CLI="1"
            shift
            ;;
        --api_key)
            API_KEY_CLI="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            if [ "$#" -gt 0 ]; then
                EXTRA_ARGS+=("$@")
            fi
            break
            ;;
        -*)
            # Unknown flag: forward remaining args to Python unchanged.
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            # Backward-compatible positional API key.
            if [ -z "$API_KEY_CLI" ]; then
                API_KEY_CLI="$1"
            else
                EXTRA_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

# Resolve values with explicit precedence: CLI > ENV > default.
VIDEO_PATH="${VIDEO_PATH_CLI:-${VIDEO_PATH:-assets/videos/chinacreekclipped.mp4}}"
PROMPT="${PROMPT_CLI:-${PROMPT:-Identify and segment small creatures in the underwater scene.}}"
OUTPUT_ROOT="${OUTPUT_ROOT_CLI:-${OUTPUT_DIR:-sam3_video_agent_out}}"
RUN_NAME="${RUN_NAME_CLI:-run_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_ROOT%/}/${RUN_NAME}"
MODEL="${MODEL_CLI:-${GEMINI_MODEL:-gemini-3.0-flash}}"
ANALYSIS_SECOND_RESOLVED="${ANALYSIS_SECOND_CLI:-${ANALYSIS_SECOND:-0}}"
GEMINI_API_MODE_RESOLVED="${GEMINI_API_MODE_CLI:-${GEMINI_API_MODE:-auto}}"
GEMINI_SDK_RESOLVED="${GEMINI_SDK_CLI:-${SAM3_GEMINI_SDK:-auto}}"
GEMINI_USE_SYSTEM_INSTRUCTION_RESOLVED="${GEMINI_USE_SYSTEM_INSTRUCTION_CLI:-${GEMINI_USE_SYSTEM_INSTRUCTION:-1}}"
GEMINI_TIMEOUT_SEC_RESOLVED="${GEMINI_TIMEOUT_SEC_CLI:-${GEMINI_TIMEOUT_SEC:-}}"
GEMINI_DEBUG_DUMP_RESOLVED="${GEMINI_DEBUG_DUMP_CLI:-${GEMINI_DEBUG_DUMP:-0}}"
GEMINI_DEBUG_DIR_RESOLVED="${GEMINI_DEBUG_DIR_CLI:-${GEMINI_DEBUG_DIR:-${OUTPUT_DIR}/gemini_debug}}"
GEMINI_TEXT_BEFORE_IMAGE_RESOLVED="${GEMINI_TEXT_BEFORE_IMAGE_CLI:-${GEMINI_TEXT_BEFORE_IMAGE:-1}}"
SAVE_PROMPTS_RESOLVED="${SAVE_PROMPTS_CLI:-${SAVE_PROMPTS:-0}}"

mkdir -p "$OUTPUT_DIR"
export SAM3_GEMINI_TEXT_BEFORE_IMAGE="${GEMINI_TEXT_BEFORE_IMAGE_RESOLVED}"

echo "Running Gemini Video Agent Test..."
echo "Video: ${VIDEO_PATH}"
echo "Gemini model: ${MODEL} | mode: ${GEMINI_API_MODE_RESOLVED} | sdk: ${GEMINI_SDK_RESOLVED}"
echo "Analysis second: ${ANALYSIS_SECOND_RESOLVED}"
echo "Output dir: ${OUTPUT_DIR}"

CMD=(
    .venv/bin/python sam3/apps/gemini_video_agent.py
    --video_path "${VIDEO_PATH}"
    --prompt "${PROMPT}"
    --model "${MODEL}"
    --gemini_sdk "${GEMINI_SDK_RESOLVED}"
    --analysis_second "${ANALYSIS_SECOND_RESOLVED}"
    --output_dir "${OUTPUT_DIR}"
    --gemini_api_mode "${GEMINI_API_MODE_RESOLVED}"
    --gemini_use_system_instruction "${GEMINI_USE_SYSTEM_INSTRUCTION_RESOLVED}"
)
if [ -n "${GEMINI_TIMEOUT_SEC_RESOLVED}" ]; then
    CMD+=(--gemini_timeout_sec "${GEMINI_TIMEOUT_SEC_RESOLVED}")
fi
if [ "${GEMINI_DEBUG_DUMP_RESOLVED}" = "1" ]; then
    CMD+=(--gemini_debug_dump --gemini_debug_dir "${GEMINI_DEBUG_DIR_RESOLVED}")
fi
if [ "${SAVE_PROMPTS_RESOLVED}" = "1" ]; then
    CMD+=(--save_prompts)
fi
if [ -n "${API_KEY_CLI}" ]; then
    CMD+=(--api_key "${API_KEY_CLI}")
fi
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

"${CMD[@]}"

echo "Test Complete. Check ${OUTPUT_DIR} for results."
