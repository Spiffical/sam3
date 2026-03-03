# Nibi Model Comparison Bundle

This folder is a portable bundle for comparing vision-reasoning models with the
SAM3 video agent workflow on Alliance Nibi.

It is designed for:
- one-video pilot runs first,
- reproducible model-to-model comparisons,
- standardized result summaries you can paste back to Codex for interpretation.

## What Is Included

- `models.json`: model matrix and launch settings.
- `run_video_agent_openai.py`: open-model runner for SAM3 agent mode.
- `frame_quality.py`: frame-quality scan utilities for corrupt/blank-frame detection.
- `frame_quality_mllm.py`: MLLM-first per-frame validity classification (valid/invalid) across full videos.
- `keyframe_discovery.py`: motion-driven keyframe proposal for temporal agent runs.
- `keyframe_discovery_mllm.py`: MLLM temporal keyframe/event discovery over frame windows.
- `track_id_matching.py`: IoU-based object-ID assignment across keyframe updates.
- `TEMPORAL_UNDERWATER_PIPELINE_PLAN.md`: design and rollout notes for temporal underwater workflow.
- Temporal discovery prompt templates:
  - `sam3/agent/system_prompts/system_prompt_temporal_discovery_underwater.txt`
  - `sam3/agent/system_prompts/system_prompt_temporal_discovery_general.txt`
- `run_model_matrix.py`: orchestrates multiple model runs.
- `summarize_runs.py`: builds `summary.csv` and `summary.md`.
- `generate_paste_report.py`: creates `PASTE_TO_CODEX.md`.
- `slurm/nibi_single_model.sbatch`: template for one model + one video.
- `slurm/nibi_matrix_array.sbatch`: template for job arrays over video list.
- `slurm/nibi_qwen35_tp8_candidate.sbatch`: experimental TP8 script for Qwen3.5.
- `slurm/submit_nibi_job.sh`: CLI wrapper to submit all templates with args.
- `templates/paste_report_template.md`: manual report skeleton.

## Why This Layout

- Keeps all comparison artifacts in one place.
- Works with existing SAM3 code in this repo.
- Gives a consistent handoff format for analysis.

## Important Notes About Nibi And Model Fit

From your Nibi documentation:
- Cluster availability: since July 31, 2025.
- Login: `nibi.alliancecan.ca` (automation: `robot.nibi.alliancecan.ca`).
- Hardware: 134,400 CPU cores and 288 H100 GPUs.
- GPU nodes: 36 nodes with 8x H100 SXM 80GB each.
- Scratch policy: 1TB soft quota with 60-day grace.
- Internet: available from all nodes.

Operational implications for this bundle:
- Slurm templates use Nibi-style GPU requests (`--gpus-per-node=h100:<N>`), in
  line with your existing `../yolo_segmentation` jobs.
- Cache/output defaults are user-generic; no username is hardcoded.
- Run outputs should go to `/project`; avoid filling `/home`.
- Default pilot remains 4 GPUs for practical throughput/cost.

Model fit caveats:
- `moonshotai/Kimi-K2.5` deploy guidance uses TP8 examples; treat as advanced.
- `Qwen/Qwen3.5-397B-A17B` currently has an official HF repo size around
  807GB and BF16 weights, so full local deployment on 4xH100 is not realistic.
- Qwen3.5 model card examples use TP8 for both SGLang and vLLM.
- This SAM3 workflow remains valid even if model native video support differs,
  because the agent phase operates on extracted frame images.

## Recommended Pilot Sequence

1. Run a Gemini API baseline with your current script.
2. Run `Qwen/Qwen3-VL-30B-A3B-Instruct` local on Nibi with 4 GPUs.
3. Run `Qwen/Qwen2.5-VL-72B-Instruct` local on Nibi with 4 GPUs.
4. Run `moonshotai/Kimi-VL-A3B-Thinking-2506` local on Nibi.
5. Optionally test `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` on 4 GPUs.
6. Optionally test `Qwen/Qwen3.5-397B-A17B` on 8 GPUs only if you accept high
   memory/complexity risk.
7. Optionally test `moonshotai/Kimi-K2.5` on 8 GPUs.

## Quick Start

1. Edit `models.json`:
- set `enabled` flags,
- update `server_url`/`launch_cmd`,
- set API key env names you actually use.

2. Run one local matrix (manual or interactive allocation):

```bash
python nibi_model_compare/run_model_matrix.py \
  --models_file nibi_model_compare/models.json \
  --video_path /path/to/onc_pilot.mp4 \
  --prompt "segment all visible marine organisms" \
  --output_root nibi_model_compare/runs/pilot_01 \
  --gpus 0,1,2,3
```

3. Submit via Slurm wrapper (all key knobs are CLI args):

```bash
bash nibi_model_compare/slurm/submit_nibi_job.sh \
  --template single \
  --account <your-account> \
  --video-path /project/<your-account>/$USER/data/onc/chinacreekclipped.mp4 \
  --prompt "identify and segment small creatures in the underwater scene" \
  --image-size 1008 \
  --gpus-per-node h100:2 \
  --tp-size 1 \
  --vllm-cuda-visible-devices 0 \
  --runner-cuda-visible-devices 1 \
  --runner-gpu-ids 0 \
  --max-completion-tokens 512 \
  --debug
```

Defaults used by templates/wrapper:
- `PROJECT_ROOT` defaults to `REPO_ROOT`.
- `DEFAULT_PROJECT_PREFIX` defaults to `/project/${SLURM_ACCOUNT:-${ACCOUNT:-$USER}}/$USER`.
- `PROJECT_CACHE_ROOT` defaults to `${DEFAULT_PROJECT_PREFIX}/hf-cache`.
- `IMAGE_SIZE` defaults to `1008` for the current SAM3 video checkpoint.
- Temporary per-job staging is under `$SLURM_TMPDIR` when available.
- Slurm resources default to the current template profile (`single`, `array`, `tp8`) and can be overridden with CLI args.
- Any extra env variable can be forwarded with `--set-env KEY=VALUE`.

4. Review outputs:
- `nibi_model_compare/runs/pilot_01/summary.csv`
- `nibi_model_compare/runs/pilot_01/summary.md`
- `nibi_model_compare/runs/pilot_01/PASTE_TO_CODEX.md`

5. Copy and paste `PASTE_TO_CODEX.md` back to Codex for interpretation.

## Output Structure

Each model gets one folder under `output_root`:

```text
<output_root>/
  <model_key>/
    frame_0.jpg
    generated_prompts.json
    output_video.mp4
    run_metrics.json
    launcher_metrics.json
    agent_out/
    sam_service/
  summary.csv
  summary.md
  PASTE_TO_CODEX.md
```

When using Slurm templates, outputs are written in two phases:
1. Stage artifacts in `$SLURM_TMPDIR`.
2. Copy finalized results to your configured `OUT_ROOT` (usually under `/project/...`).

## Sources Used For Planning (Primary)

- https://docs.alliancecan.ca/wiki/Nibi
- https://huggingface.co/moonshotai/Kimi-K2.5
- https://huggingface.co/moonshotai/Kimi-K2.5/blob/main/docs/deploy_guidance.md
- https://huggingface.co/moonshotai/Kimi-VL-A3B-Thinking-2506
- https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct/blob/main/README.md
- https://huggingface.co/Qwen/Qwen3.5-397B-A17B
- https://huggingface.co/collections/Qwen/qwen35
- https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct
- https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-FP8
- https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct-FP8
- https://docs.vllm.ai/en/latest/models/supported_models.html
- https://deepmind.google/models/gemini/
- https://videommmu.github.io/
- https://helpwiki.sharcnet.ca/wiki/images/a/ac/Migration_webinar_2025.pdf
- https://sharcnet.github.io/
- https://www.top500.org/system/180370/
