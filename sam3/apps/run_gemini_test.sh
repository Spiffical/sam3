#!/bin/bash

# Ensure we are in the root directory if running from there, or handle paths
# Script location
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$(dirname "$DIR")")"
cd "$ROOT_DIR"

echo "Running Gemini Video Agent Test..."
echo "Video: assets/videos/chinacreekclipped.mp4"

# Pass first argument as API key if provided, otherwise assume it's in .env or vars
EXTRA_ARGS=""
if [ ! -z "$1" ]; then
    EXTRA_ARGS="--api_key $1"
fi

.venv/bin/python sam3/apps/gemini_video_agent.py \
    --video_path assets/videos/chinacreekclipped.mp4 \
    --prompt "Identify and segment small creatures in the underwater scene." \
    --output_dir sam3_video_agent_out \
    --save_prompts \
    $EXTRA_ARGS

echo "Test Complete. Check sam3_video_agent_out for results."
