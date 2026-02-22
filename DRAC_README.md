# DRAC / Alliance Quickstart For SAM3 + vLLM (Nibi)

This README captures the working setup used in this repo for running SAM3 video-agent tests on Alliance resources (especially Nibi).

## Scope

- Build a clean `.venv` on Alliance clusters.
- Install dependencies in a way that avoids common Alliance wheelhouse/OpenCV issues.
- Run an interactive end-to-end test (`vllm serve` + `run_video_agent_openai.py`).
- Avoid `/home` quota failures by using `$SLURM_TMPDIR` and `/project` caches.

## TL;DR Copy/Paste (2-GPU Interactive, Recommended)

Use this when you already have an interactive Nibi allocation with `h100:2`.
It runs `vllm` on GPU0 and SAM3 tracking on GPU1 to avoid OOM.

```bash
cd ~/sam3
source .venv/bin/activate
set -a; source .env; set +a

export ACCOUNT="${ACCOUNT:-rpp-kmoran}"
export MODEL_ID="Qwen/Qwen3-VL-30B-A3B-Instruct"
export PORT=8001
export VIDEO_PATH="/project/${ACCOUNT}/${USER}/data/onc/chinacreekclipped.mp4"
export OUT_DIR="/project/${ACCOUNT}/${USER}/sam3/runs/interactive_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"

# Keep all HF/PyTorch cache off $HOME
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

pkill -f "vllm serve" || true

# GPU0: vLLM server
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_ID" \
  --tensor-parallel-size 1 \
  --allowed-local-media-path / \
  --gpu-memory-utilization 0.92 \
  --max-model-len 16384 \
  --max-num-seqs 1 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --port "$PORT" >"/tmp/vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

# Wait for server readiness and fail fast if it dies
for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM failed to start. Last log lines:"
    tail -n 120 "/tmp/vllm_${SLURM_JOB_ID}.log"
    exit 1
  fi
  sleep 2
done
curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null

# GPU1: SAM3 agent + propagation
CUDA_VISIBLE_DEVICES=1 \
SAM3_DISABLE_WARMUP=1 \
SAM3_MAX_IMAGES_PER_REQUEST=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python nibi_model_compare/run_video_agent_openai.py \
  --video_path "$VIDEO_PATH" \
  --prompt "identify and segment small creatures in the underwater scene" \
  --server_url "http://127.0.0.1:${PORT}/v1" \
  --model "$MODEL_ID" \
  --output_dir "$OUT_DIR" \
  --gpus 0 \
  --image_size 1008 \
  --max_completion_tokens 1024 \
  --debug 2>&1 | tee "/tmp/sam3_runner_${SLURM_JOB_ID}.log"

cat "$OUT_DIR/run_metrics.json"
echo "Output video: $OUT_DIR/output_video.mp4"
echo "Prompts JSON: $OUT_DIR/generated_prompts.json"

kill "$VLLM_PID" 2>/dev/null || true
```

## TL;DR Copy/Paste (Submit Batch Job)

If you prefer `sbatch` instead of interactive runs:

```bash
cd ~/sam3
export ACCOUNT="${ACCOUNT:-rpp-kmoran}"

nibi_model_compare/slurm/submit_nibi_job.sh \
  --template single \
  --account "$ACCOUNT" \
  --gpus-per-node h100:2 \
  --cpus-per-task 16 \
  --mem 128000M \
  --time 02:00:00 \
  --tp-size 1 \
  --vllm-cuda-visible-devices 0 \
  --runner-cuda-visible-devices 1 \
  --runner-gpu-ids 0 \
  --max-model-len 16384 \
  --image-size 1008 \
  --max-completion-tokens 1024 \
  --video-path "/project/${ACCOUNT}/${USER}/data/onc/chinacreekclipped.mp4" \
  --prompt "identify and segment small creatures in the underwater scene" \
  --debug \
  --set-env SAM3_DISABLE_WARMUP=1 \
  --set-env SAM3_MAX_IMAGES_PER_REQUEST=1 \
  --set-env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 1) Interactive Allocation

Set your Alliance account once:

```bash
export ACCOUNT="${ACCOUNT:-rpp-kmoran}"
```

Quick single-GPU smoke test:

```bash
salloc --account="$ACCOUNT" --nodes=1 --gpus-per-node=h100:1 --cpus-per-task=8 --mem=64G --time=01:00:00
```

Larger-context test (recommended for Qwen3-VL-30B):

```bash
salloc --account="$ACCOUNT" --nodes=1 --gpus-per-node=h100:2 --cpus-per-task=16 --mem=128G --time=02:00:00
```

Notes:
- Do not use MIG profiles for this workload.
- On Nibi, `partition=gpu` is typically not needed for these interactive commands.

## 2) Modules And `.venv`

Important: load `opencv` module before activating/using the virtual environment.

```bash
cd ~/sam3
deactivate 2>/dev/null || true

module purge
module load StdEnv/2023
module load python/3.11 cuda cudnn
module load scipy-stack
module load opencv/4.12.0

python -m venv .venv
source .venv/bin/activate

export ACCOUNT="${ACCOUNT:-rpp-kmoran}"
export PYTHONNOUSERSITE=1
export PIP_NO_USER=1
export PIP_CACHE_DIR="/project/${ACCOUNT}/${USER}/pip-cache"
mkdir -p "$PIP_CACHE_DIR"
```

## 3) Install Dependencies

### 3.1 Base tooling

```bash
python -m pip --isolated install --no-index --upgrade pip setuptools wheel packaging
```

### 3.2 SAM3 package (editable)

```bash
python -m pip --isolated install --no-index -e . || python -m pip install -e .
python -m pip --isolated install --no-index -e ".[notebooks,frontend,train]" || python -m pip install -e ".[notebooks,frontend,train]"
```

### 3.3 Runtime packages

Try normal Alliance wheelhouse install first:

```bash
python -m pip --isolated install --no-index vllm openai pycocotools || true
```

If `opencv-noinstall` / `opencv-python-headless` fails during `vllm` dependency resolution, use:

```bash
python -m pip --isolated install --no-index vllm --no-deps
python -m pip --isolated install --no-index openai pycocotools packaging setuptools
```

Optional: install additional `vllm` deps while skipping problematic OpenCV deps:

```bash
python - <<'PY' > /tmp/vllm_reqs_nibi.txt
from importlib.metadata import requires
from packaging.requirements import Requirement

skip = {"opencv-python-headless", "opencv-noinstall"}
for raw in requires("vllm") or []:
    req = Requirement(raw)
    if req.marker and not req.marker.evaluate():
        continue
    if req.name.lower() in skip:
        continue
    print(str(req))
PY

python -m pip --isolated install --no-index -r /tmp/vllm_reqs_nibi.txt || true
```

## 4) Verify Environment

```bash
python - <<'PY'
import torch, cv2, vllm, openai, pycocotools
print("OK  torch")
print("OK  cv2")
print("OK  vllm")
print("OK  openai")
print("OK  pycocotools")
PY
```

## 5) Hugging Face Token And Access

Put token in `.env`:

```bash
HF_TOKEN=hf_xxx
```

Load it:

```bash
set -a; source .env; set +a
```

You must have access approved for:
- `facebook/sam3` (SAM3 weights/config used by this workflow)

## 6) Cache Layout (Avoid Disk Quota Errors)

Use node-local cache during the job:

```bash
export ACCOUNT="${ACCOUNT:-rpp-kmoran}"
export JOB_CACHE_ROOT="${SLURM_TMPDIR:-/project/${ACCOUNT}/${USER}/hf-cache/tmp}/hf-cache"
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
```

Optional cleanup of stale home-cache model download:

```bash
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct*
```

## 7) One-Shell Interactive Test

This runs `vllm` in background and the SAM3 script in the same shell.

```bash
cd ~/sam3
source .venv/bin/activate
set -a; source .env; set +a

export ACCOUNT="${ACCOUNT:-rpp-kmoran}"
export MODEL_ID="Qwen/Qwen3-VL-30B-A3B-Instruct"
export PORT=8001
export SAM3_IMAGE_DETAIL=low
export SAM3_AGENT_IMAGE_MAX_EDGE=768

# Choose TP size based on visible GPUs (max 2 for this test)
TP_SIZE=$(nvidia-smi -L | wc -l | tr -d ' ')
if [ "$TP_SIZE" -gt 2 ]; then TP_SIZE=2; fi

vllm serve "$MODEL_ID" \
  --tensor-parallel-size "$TP_SIZE" \
  --allowed-local-media-path / \
  --gpu-memory-utilization 0.92 \
  --max-model-len 32768 \
  --max-num-seqs 1 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --port "$PORT" > /tmp/vllm_run.log 2>&1 &
VLLM_PID=$!

python - <<'PY'
import time, urllib.request
url = "http://127.0.0.1:8001/v1/models"
for _ in range(300):
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            if r.status == 200:
                print("vLLM ready")
                raise SystemExit(0)
    except Exception:
        time.sleep(2)
print("vLLM not ready; check /tmp/vllm_run.log")
raise SystemExit(1)
PY

VIDEO_PATH="/project/${ACCOUNT}/${USER}/data/onc/chinacreekclipped.mp4"
OUT_DIR="/project/${ACCOUNT}/${USER}/sam3/runs/interactive_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"

python nibi_model_compare/run_video_agent_openai.py \
  --video_path "$VIDEO_PATH" \
  --prompt "segment all visible marine organisms" \
  --server_url "http://127.0.0.1:${PORT}/v1" \
  --model "$MODEL_ID" \
  --output_dir "$OUT_DIR" \
  --gpus 0 \
  --max_completion_tokens 256 \
  --debug

cat "$OUT_DIR/run_metrics.json"
kill "$VLLM_PID" 2>/dev/null || true
```

## 8) Common Failures And Fixes

- `No module named 'decord'`:
  - Install SAM3 extras (`.[notebooks,frontend,train]`) or use latest repo patches where decord fallbacks are handled.
- `No module named 'gradio'`:
  - Install SAM3 extras.
- `No module named 'pkg_resources'`:
  - Use latest repo patch removing hard dependency on `pkg_resources` in model builder path.
- `No module named 'bioclip'` during CLI run:
  - Use latest repo patch where interactive-video package/UI imports are lazy/optional.
- `Could not find bpe_simple_vocab_16e6.txt.gz`:
  - Latest script resolves `sam3/assets/...`; optionally set:
    - `export SAM3_BPE_PATH=/home/$USER/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz`
- `World size (2) > available GPUs (1)`:
  - Lower `--tensor-parallel-size` to visible GPU count, or request more GPUs in `salloc`.
- `Disk quota exceeded`:
  - Ensure all HF caches are on `$SLURM_TMPDIR` or `/project`, not `/home`.
- `--limit-mm-per-prompt` parse error:
  - Use JSON format:
    - `--limit-mm-per-prompt '{"image":1,"video":0}'`
- Context error (`input+output > context length`):
  - Reduce `--max_completion_tokens` (start with 256).
  - Keep `SAM3_IMAGE_DETAIL=low`, `SAM3_AGENT_IMAGE_MAX_EDGE=768`.
  - Increase `--max-model-len` only after model load is stable.

## 9) Notes On Slurm Templates In This Repo

Templates in `nibi_model_compare/slurm` are configured to:
- load `.env` automatically (if present),
- map `HF_TOKEN` into `HUGGINGFACE_HUB_TOKEN` if needed,
- use `$SLURM_TMPDIR`-backed cache by default with optional `/project` prewarm/sync.

## 10) Security

- Do not commit tokens to git.
- If any PAT/token was exposed in logs/chats, rotate it immediately.
