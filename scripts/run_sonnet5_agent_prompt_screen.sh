#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [output-root]" >&2
  exit 2
fi

output_root="${1:-runs/presentation_benchmark/seatube_meagan_v1/sonnet5_prompt_screen/20260817_v1}"

run_case() {
  local case_id="$1"
  local frame_id="$2"
  local prompt="$3"
  local case_root="${output_root}/${case_id}"

  if [[ -f "${case_root}/benchmark_metadata.json" ]] && \
    jq -e '.runs[-1].status == "failed"' \
      "${case_root}/benchmark_metadata.json" >/dev/null 2>&1; then
    echo "[$case_id] previously failed; keeping failure evidence and continuing"
    return 0
  fi

  if ! PYTHONPATH=. .venv/bin/python scripts/run_fixed_frame_agent_benchmark.py \
    --output-dir "$case_root" \
    --model claude-sonnet-5 \
    --repeats 1 \
    --prompt "$prompt" \
    --prompt-profile underwater \
    --max-completion-tokens 2048 \
    --max-generations 20 \
    --confidence-threshold 0.40 \
    --frame-id "$frame_id" \
    --resume; then
    echo "[$case_id] failed; continuing prompt screen" >&2
    return 0
  fi

  local overlay="${case_root}/repeat_1/${frame_id}/overlay.mp4"
  local preview="${case_root}/repeat_1/${frame_id}/presentation_overlay.png"
  if [[ -f "$overlay" && ! -f "$preview" ]]; then
    ffmpeg -loglevel error -y -i "$overlay" -frames:v 1 "$preview"
  fi
}

# Positive-control crowded scene: exact WoRMS taxa known to occur in the clip.
run_case \
  "03_taxa" \
  "03_anemone_tubeworm_field" \
  "Actiniaria, Lamellibrachia, and Decapoda"

# Dense thicket: generic, morphology-rich, and exact-taxonomy variants.
run_case \
  "04_all_marine_life" \
  "04_dense_coral_thicket" \
  "all visible marine life"
run_case \
  "04_sessile_morphology" \
  "04_dense_coral_thicket" \
  "corals, sea fans, sea whips, sponges, anemones, and other sessile marine life"
run_case \
  "04_taxa" \
  "04_dense_coral_thicket" \
  "Amphianthus, Comatulida, Ophiacanthidae, and Parazoanthidae"

# Wide garden: the same generic/morphology controls and its exact WoRMS taxa.
run_case \
  "05_all_marine_life" \
  "05_wide_coral_garden" \
  "all visible marine life"
run_case \
  "05_sessile_morphology" \
  "05_wide_coral_garden" \
  "corals, sea fans, sea whips, sponges, anemones, and other sessile marine life"
run_case \
  "05_taxa" \
  "05_wide_coral_garden" \
  "Acanthogorgia, Caulophacus, Chrysogorgia, Euryalidae, Hemicorallium, Paragorgia, Farreidae, Narella, Parazoanthidae, and Walteria"
