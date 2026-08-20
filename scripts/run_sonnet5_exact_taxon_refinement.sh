#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [output-root]" >&2
  exit 2
fi

output_root="${1:-runs/presentation_benchmark/seatube_meagan_v1/sonnet5_prompt_screen/20260817_exact_taxa_v1}"

run_case() {
  local case_id="$1"
  local frame_id="$2"
  local prompt="$3"
  local case_root="${output_root}/${case_id}"

  if ! PYTHONPATH=. .venv/bin/python scripts/run_fixed_frame_agent_benchmark.py \
    --output-dir "$case_root" \
    --model claude-sonnet-5 \
    --repeats 1 \
    --prompt "$prompt" \
    --prompt-profile underwater \
    --max-completion-tokens 2048 \
    --max-generations 12 \
    --confidence-threshold 0.40 \
    --frame-id "$frame_id" \
    --resume; then
    echo "[$case_id] failed; continuing exact-taxon refinement" >&2
    return 0
  fi

  local overlay="${case_root}/repeat_1/${frame_id}/overlay.mp4"
  local preview="${case_root}/repeat_1/${frame_id}/presentation_overlay.png"
  if [[ -f "$overlay" && ! -f "$preview" ]]; then
    ffmpeg -loglevel error -y -i "$overlay" -frames:v 1 "$preview"
  fi
}

run_case \
  "03_actiniaria_exact" \
  "03_anemone_tubeworm_field" \
  "First call segment_phrase with exactly 'Actiniaria'. Inspect those masks, keep valid distinct organisms, and then return."

run_case \
  "05_walteria_exact" \
  "05_wide_coral_garden" \
  "First call segment_phrase with exactly 'Walteria'. Inspect those masks, keep valid distinct organisms, and then return."

run_case \
  "05_chrysogorgia_exact" \
  "05_wide_coral_garden" \
  "First call segment_phrase with exactly 'Chrysogorgia'. Inspect those masks, keep valid distinct organisms, and then return."

run_case \
  "05_walteria_chrysogorgia_exact" \
  "05_wide_coral_garden" \
  "Call segment_phrase separately with exactly 'Walteria' and then exactly 'Chrysogorgia'. Inspect the accumulated masks, keep every valid distinct organism, remove duplicates, and then return."
