#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [output-root]" >&2
  exit 2
fi

output_root="${1:-runs/presentation_benchmark/seatube_meagan_v1/sonnet5_prompt_validation/20260817_v1}"

run_case() {
  local case_id="$1"
  local frame_id="$2"
  local prompt="$3"
  local max_generations="$4"
  local case_root="${output_root}/${case_id}"

  if ! PYTHONPATH=. .venv/bin/python scripts/run_fixed_frame_agent_benchmark.py \
    --output-dir "$case_root" \
    --model claude-sonnet-5 \
    --repeats 3 \
    --prompt "$prompt" \
    --prompt-profile underwater \
    --max-completion-tokens 2048 \
    --max-generations "$max_generations" \
    --confidence-threshold 0.40 \
    --frame-id "$frame_id" \
    --resume; then
    echo "[$case_id] failed; retaining evidence and continuing validation" >&2
    return 0
  fi

  local repeat overlay preview
  for repeat in 1 2 3; do
    overlay="${case_root}/repeat_${repeat}/${frame_id}/overlay.mp4"
    preview="${case_root}/repeat_${repeat}/${frame_id}/presentation_overlay.png"
    if [[ -f "$overlay" && ! -f "$preview" ]]; then
      ffmpeg -loglevel error -y -i "$overlay" -frames:v 1 "$preview"
    fi
  done
}

# Frame 03: compare an unconstrained known-taxa cue with exact taxon forcing.
run_case \
  "03_known_taxa" \
  "03_anemone_tubeworm_field" \
  "Actiniaria, Lamellibrachia, and Decapoda" \
  20

# Video review revealed a camouflaged fish-like animal that `fish` misses but
# `small fish` retrieves.  Both proposal calls must finish before review.
run_case \
  "03_sea_anemone_small_fish_union" \
  "03_anemone_tubeworm_field" \
  "Call segment_phrase separately with exactly 'sea anemone' and then exactly 'small fish'. Keep masks accumulated across both calls. Inspect every mask. Keep every visible biological organism, including anemones, the fish-like animal in the left crevice, and plausible tube-worm plumes. Reject logo and substrate artifacts, remove overlapping or partial duplicates, and return one mask per distinct organism." \
  20

run_case \
  "03_actiniaria_exact" \
  "03_anemone_tubeworm_field" \
  "First call segment_phrase with exactly 'Actiniaria'. Inspect those masks, keep valid distinct organisms, and then return." \
  12

# Frame 05: exact Walteria was the only useful text-retrieval handle in screening.
run_case \
  "05_walteria_exact" \
  "05_wide_coral_garden" \
  "First call segment_phrase with exactly 'Walteria'. Inspect those masks, keep valid distinct organisms, and then return." \
  12
