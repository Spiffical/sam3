#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [output-root]" >&2
  exit 2
fi

output_root="${1:-runs/presentation_benchmark/seatube_meagan_v1/custom_flow/20260817_taxon_text_fable_persistent2_batch_repeats3}"
firstpass_base="runs/presentation_benchmark/seatube_meagan_v1/sam3_agent/20260816_sonnet46_repeats3"
mask_guided_passes="${SAM3_MASK_GUIDED_PASSES:-2}"
sparse_pass_args=()
if [[ "${SAM3_SPARSE_EXTRA_PASS:-1}" == "1" ]]; then
  sparse_pass_args+=(--sparse-extra-pass)
fi
frame_args=()
if [[ -n "${SAM3_FRAME_IDS:-}" ]]; then
  IFS=',' read -r -a requested_frames <<< "$SAM3_FRAME_IDS"
  for frame_id in "${requested_frames[@]}"; do
    frame_args+=(--frame-id "$frame_id")
  done
fi

for repeat in 1 2 3; do
  repeat_output="${output_root}/repeat_${repeat}"
  mkdir -p "$repeat_output"
  PYTHONPATH=. .venv/bin/python scripts/run_presentation_custom_flow.py \
    --manifest configs/seatube_meagan_five_frames_v1.json \
    --firstpass-root "${firstpass_base}/repeat_${repeat}" \
    --output-dir "$repeat_output" \
    --model claude-fable-5 \
    --firstpass-model claude-sonnet-4-6 \
    --text-proposal-mode compact \
    --text-proposal-threshold 0.45 \
    --min-text-confidence 0.70 \
    --finder-mode mask-guided \
    --temporal-seconds 0.5,1.0 \
    --mask-guided-passes "$mask_guided_passes" \
    --border-scan off \
    --min-recovery-confidence 0.80 \
    "${sparse_pass_args[@]}" \
    "${frame_args[@]}" \
    --resume
done

PYTHONPATH=. .venv/bin/python scripts/summarize_custom_flow_repeats.py "$output_root"
