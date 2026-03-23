# FathomNet 2026 Kaggle Workspace

This package is the dedicated home for the `fathomnet-2026` Kaggle competition.

The goal is to keep competition work clearly separated from:

- ONC-specific underwater video experiments
- temporal tracking and ID reassignment utilities
- generic SAM3 examples and demos

## Package Layout

- `categories.py`: validates the competition's 32-category taxonomy from COCO JSON.
- `dataset.py`: loads COCO metadata and writes image manifests.
- `prompts.py`: competition-specific zero-shot detection and crop-classification prompts.
- `submission.py`: validates prediction records and writes Kaggle submission CSVs.
- `runtime.py`: loads the SAM3 + Qwen runtime pieces used by the zero-shot runner.
- `zero_shot.py`: runs the end-to-end SAM3 proposal + Qwen crop-classification pipeline.
- `bundle.py`: manages Kaggle downloads, extracted project-data layout, bundle creation, and split staging.

## Scripts

The matching entry points live under `scripts/fathomnet_2026/`.

- `download_and_bundle_data.py`: download Kaggle data into project storage and build a compressed dataset bundle.
- `extract_dataset_bundle.py`: unpack the prepared bundle into a job-local directory and write a staging manifest.
- `run_zero_shot_submission.py`: run zero-shot inference over a chosen split and write `submission.csv`.
- `prepare_dataset_manifest.py`: write a simple image manifest from a COCO dataset JSON.
- `write_submission_csv.py`: validate predictions and write Kaggle-formatted CSV output.

The Nibi submit wrapper lives under `nibi_model_compare/`.

- `submit_fathomnet_2026_zero_shot.sh`
- `slurm/fathomnet_2026_zero_shot.sbatch`

## Zero-Shot Flow

1. Load `dataset_test.json` and the extracted test images.
2. Run SAM3 image-mode proposal generation with the `fathomnet_2026` agent prompt profile.
3. Merge and deduplicate proposals across one broad prompt plus rescue prompts.
4. Ask Qwen 3.5 to classify each crop into exactly one of the 32 competition labels or drop it.
5. Apply final per-category NMS and export Kaggle submission rows.

## Nibi Data Layout

The shared project-storage layout defaults to:

```text
/project/rpp-kmoran/merileo/data/fathomnet_2026_kaggle/
  downloads/
  expanded/
  manifests/
  bundles/
```

The intended workflow is:

1. Download Kaggle files into `downloads/`.
2. Expand them into `expanded/`.
3. Write split manifests into `manifests/`.
4. Create a compressed bundle in `bundles/`.
5. During a Slurm job, extract that bundle into `$SLURM_TMPDIR`.

This keeps the canonical dataset archive on project storage while using fast local scratch for active jobs.

## Nibi Quickstart

### 1. Pull the branch and activate your env

```bash
git checkout codex/fathomnet-kaggle-setup
source "$HOME/sam3/.venv-qwen35/bin/activate"
```

### 2. Download and bundle the Kaggle data

Make sure your Kaggle credentials are available on the login node, then run:

```bash
python3 scripts/fathomnet_2026/download_and_bundle_data.py \
  --project-data-root /project/rpp-kmoran/merileo/data \
  --dataset-subdir fathomnet_2026_kaggle
```

By default this writes:

- bundle: `/project/rpp-kmoran/merileo/data/fathomnet_2026_kaggle/bundles/fathomnet_2026_kaggle.tar.zst`
- metadata: `/project/rpp-kmoran/merileo/data/fathomnet_2026_kaggle/bundles/fathomnet_2026_kaggle.metadata.json`

### 3. Launch the zero-shot job

This wrapper stages the bundle into `$SLURM_TMPDIR`, starts vLLM for Qwen 3.5, and runs the competition pipeline:

```bash
cmd=(
  nibi_model_compare/submit_fathomnet_2026_zero_shot.sh
  --account def-kmoran
  --venv-path "$HOME/sam3/.venv-qwen35"
  --vllm-runtime apptainer
  --apptainer-image "${SCRATCH:-/scratch/$USER}/vllm-openai-nightly.sif"
  --model-id "Qwen/Qwen3.5-27B"
  --gpus-per-node h100:2
  --vllm-cuda-visible-devices 0
  --runner-cuda-visible-devices 1
  --max-completion-tokens 1024
  --debug
)
"${cmd[@]}"
```

Useful extra flags:

- `--dataset-split test` to run the Kaggle test split
- `--max-images 25` for a smoke test
- `--compile-image-model` to compile the SAM3 image model
- `--dry-run` to print the final `sbatch` command without submitting

## Outputs

Each run writes:

- `submission.csv`
- `predictions.jsonl`
- `raw_candidates.jsonl`
- `summary.json`
- optional per-image debug artifacts when `--debug` is enabled

Slurm outputs land under `$SCRATCH`, while the canonical dataset bundle stays under `/project/rpp-kmoran/merileo/data`.

## Notes

- The bundle script defaults to `.tar.zst`, which is the intended format for Nibi.
- If you need a local fallback on a machine without `zstd`, pass `--bundle-filename fathomnet_2026_kaggle.tar.gz`.
- The submit wrapper expects Hugging Face credentials to be available through your `.env` file or environment variables when the job starts.
