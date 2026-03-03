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
  --prompt-profile underwater \
  --server_url "http://127.0.0.1:${PORT}/v1" \
  --model "$MODEL_ID" \
  --output_dir "$OUT_DIR" \
  --gpus 0 \
  --image_size 1008 \
  --temporal_keyframe_pipeline \
  --discovery_mode hybrid \
  --mllm_discovery_window_size 4 \
  --mllm_discovery_window_stride 24 \
  --mllm_discovery_max_json_retries 2 \
  --max_keyframes 6 \
  --min_keyframe_gap 24 \
  --invalid_frame_source mllm \
  --mllm_invalid_window_size 4 \
  --mllm_invalid_window_stride 4 \
  --drop_invalid_frames \
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
  --prompt-profile underwater \
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
python -m pip --isolated install --no-index -e . || PIP_CONFIG_FILE=/dev/null python -m pip install -i https://pypi.org/simple -e .
python -m pip --isolated install --no-index -e ".[notebooks,frontend,train]" || PIP_CONFIG_FILE=/dev/null python -m pip install -i https://pypi.org/simple -e ".[notebooks,frontend,train]"
```

### 3.3 Runtime packages

Try normal Alliance wheelhouse install first:

```bash
python -m pip --isolated install --no-index vllm openai pycocotools timm ftfy scikit-image scikit-learn pandas matplotlib || true
```

If `opencv-noinstall` / `opencv-python-headless` fails during `vllm` dependency resolution, use:

```bash
python -m pip --isolated install --no-index vllm --no-deps
python -m pip --isolated install --no-index openai pycocotools timm ftfy scikit-image scikit-learn pandas matplotlib packaging setuptools
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

Note:
- Alliance wheelhouse currently provides `iopath 0.1.9`, so `sam3` uses `iopath>=0.1.9` for DRAC compatibility.
- Do **not** export `PIP_CONFIG_FILE=/dev/null` globally when using `--no-index`; that hides Alliance wheelhouse links.
- If fallback is needed, use `PIP_CONFIG_FILE=/dev/null` only inline on that fallback command.
- If `grep -n "iopath>=" pyproject.toml` still shows `0.1.10`, update your checkout first.

### 3.4 Qwen3.5 compatibility stack (optional, separate venv recommended)

Qwen3.5 models may require newer `transformers`/`vllm` than Alliance wheelhouse currently provides.
Keep your existing `.venv` stable and create a second environment for Qwen3.5 tests.

```bash
cd ~/sam3
deactivate 2>/dev/null || true

module purge
module load StdEnv/2023
module load python/3.11 cuda cudnn
module load scipy-stack
module load opencv/4.12.0

python -m venv .venv-qwen35
source .venv-qwen35/bin/activate

export ACCOUNT="${ACCOUNT:-rpp-kmoran}"
export PYTHONNOUSERSITE=1
export PIP_NO_USER=1
export PIP_CACHE_DIR="/project/${ACCOUNT}/${USER}/pip-cache"
mkdir -p "$PIP_CACHE_DIR"

# Keep Alliance wheelhouse config for --no-index commands:
unset PIP_CONFIG_FILE
unset PIP_USER

# Force wheelhouse links explicitly for reliability on all nodes:
WHEELHOUSE_ARGS=(
  -f /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v4
  -f /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v3
  -f /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/generic
  -f /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic
)

# Follow the same "no-index first, fallback second" pattern.
python -m pip --isolated install --no-index "${WHEELHOUSE_ARGS[@]}" --upgrade pip setuptools wheel packaging || PIP_CONFIG_FILE=/dev/null python -m pip install -i https://pypi.org/simple --upgrade pip setuptools wheel packaging
python -m pip --isolated install --no-index "${WHEELHOUSE_ARGS[@]}" -e . --no-build-isolation --no-deps || PIP_CONFIG_FILE=/dev/null python -m pip install -i https://pypi.org/simple -e .
python -m pip --isolated install --no-index "${WHEELHOUSE_ARGS[@]}" -e ".[notebooks,frontend,train]" --no-build-isolation --no-deps || PIP_CONFIG_FILE=/dev/null python -m pip install -i https://pypi.org/simple -e ".[notebooks,frontend,train]"

# Avoid Alliance dummy OpenCV package breakage:
python -m pip --isolated install --no-index "${WHEELHOUSE_ARGS[@]}" vllm --no-deps
python -m pip --isolated install --no-index "${WHEELHOUSE_ARGS[@]}" openai pycocotools timm ftfy scikit-image scikit-learn pandas matplotlib packaging setuptools
python -m pip --isolated install --no-index "${WHEELHOUSE_ARGS[@]}" --upgrade transformers || true

python - <<'PY' > /tmp/vllm_reqs_nibi_qwen35.txt
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

python -m pip --isolated install --no-index "${WHEELHOUSE_ARGS[@]}" -r /tmp/vllm_reqs_nibi_qwen35.txt || true

# Qwen3.5 fallback (required if either condition happens):
# 1) AutoConfig fails with model type qwen3_5 / qwen3_5_moe
# 2) vLLM fails with "Model architectures ['Qwen3_5ForConditionalGeneration'] are not supported for now"
#
# Important:
# - Upgrading only transformers is not enough.
# - You must move off vllm 0.16.x for Qwen3.5.
# - Use binary-only install for vLLM so pip does not try a source build.
# - The official vLLM recipe also recommends:
#     uv pip install -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
#   (useful when uv is available on your node)
PIP_CONFIG_FILE=/dev/null python -m pip install --upgrade --force-reinstall --no-cache-dir --pre --only-binary=:all: --extra-index-url https://wheels.vllm.ai/nightly "vllm>=0.17.0.dev0"
PIP_CONFIG_FILE=/dev/null python -m pip install --upgrade --force-reinstall --no-cache-dir "transformers @ git+https://github.com/huggingface/transformers.git@main"
```

Quick compatibility check:

```bash
python - <<'PY'
from transformers import AutoConfig
for model in ("Qwen/Qwen3.5-27B", "Qwen/Qwen3.5-35B-A3B"):
    cfg = AutoConfig.from_pretrained(model)
    print(model, "->", cfg.model_type)
PY
```

If this installs `transformers>=5` and you see a `vllm requires transformers<5` warning, upgrade `vllm` to nightly in the fallback step above.

Quick vLLM runtime check for Qwen3.5:

```bash
python - <<'PY'
import vllm, transformers
print("vllm:", vllm.__version__)
print("transformers:", transformers.__version__)
PY
```

If `vllm` still prints `0.16.x`, your environment is still on the old build and Qwen3.5 will fail to start.
If binary-only install reports `No matching distribution found`, there is no compatible nightly wheel for this cluster/Python/CUDA stack; in that case Qwen3.5 is not usable with this vLLM workflow on that environment.

Nightly wheel probe (check before spending time on installs):

```bash
python - <<'PY'
import re, urllib.request
url = "https://wheels.vllm.ai/nightly/vllm/"
html = urllib.request.urlopen(url, timeout=30).read().decode()
wheels = re.findall(r'href="(vllm-[^"]+\.whl)"', html)
cp311 = [w for w in wheels if "cp311" in w and "linux_x86_64" in w]
print("cp311 linux_x86_64 wheels:", len(cp311))
for w in cp311[-10:]:
    print(w)
PY
```

If this prints zero wheels, pip will fall back to source (often fails on cluster), and you should use a containerized nightly runtime.

Container fallback (Apptainer, interactive, Slurm-safe):

```bash
# 1) Get a GPU allocation first (do NOT run on login node).
#    If you are already inside an interactive allocation or sbatch job, skip this.
salloc --account="${ACCOUNT:-rpp-kmoran}" --nodes=1 --gpus-per-node=h100:1 --cpus-per-task=8 --mem=64G --time=01:00:00

# 2) On the allocated compute node:
module load apptainer
nvidia-smi -L

export MODEL_ID="Qwen/Qwen3.5-27B"
export PORT=8006
export HF_TOKEN="${HF_TOKEN:-$HUGGINGFACE_HUB_TOKEN}"
export CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:-/scratch/$USER}/sam3-cache/${SLURM_JOB_ID:-manual}}"
export JOB_CACHE_ROOT="${JOB_CACHE_ROOT:-$CACHE_ROOT}"
export HF_HOME="${HF_HOME:-$JOB_CACHE_ROOT/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}"
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-$HF_HOME/assets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HF_HOME/xdg}"
export TMPDIR="${TMPDIR:-${SLURM_TMPDIR:-$JOB_CACHE_ROOT/tmp}}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$JOB_CACHE_ROOT/vllm}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$JOB_CACHE_ROOT/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$JOB_CACHE_ROOT/torchinductor}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$JOB_CACHE_ROOT/numba}"
export HF_HUB_DISABLE_XET=1
unset TRANSFORMERS_CACHE

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${SCRATCH:-/scratch/$USER}/apptainer-cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${SLURM_TMPDIR:-$JOB_CACHE_ROOT/apptainer-tmp}}"
export APPTAINER_HOME="${APPTAINER_HOME:-$JOB_CACHE_ROOT/apptainer-home}"
export APPTAINER_BINDPATH="${APPTAINER_BINDPATH:-$JOB_CACHE_ROOT,$APPTAINER_HOME,$APPTAINER_CACHEDIR,$APPTAINER_TMPDIR}"
# If you want to use SLURM_TMPDIR in-container, explicitly bind it:
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  APPTAINER_BINDPATH="${APPTAINER_BINDPATH},${SLURM_TMPDIR}:${SLURM_TMPDIR}"
fi
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$HF_ASSETS_CACHE" "$XDG_CACHE_HOME" "$TMPDIR" \
  "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$NUMBA_CACHE_DIR" \
  "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$APPTAINER_HOME"

# Prevent inherited host TLS settings from breaking HTTPS in container.
# These are injected into container env even when using --cleanenv.
export APPTAINERENV_HF_TOKEN="$HF_TOKEN"
export APPTAINERENV_HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export APPTAINERENV_HF_HOME="$HF_HOME"
export APPTAINERENV_HF_HUB_CACHE="$HF_HUB_CACHE"
export APPTAINERENV_HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export APPTAINERENV_HF_XET_CACHE="$HF_XET_CACHE"
export APPTAINERENV_HF_ASSETS_CACHE="$HF_ASSETS_CACHE"
export APPTAINERENV_XDG_CACHE_HOME="$XDG_CACHE_HOME"
export APPTAINERENV_TMPDIR="$TMPDIR"
export APPTAINERENV_VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT"
export APPTAINERENV_TRITON_CACHE_DIR="$TRITON_CACHE_DIR"
export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR"
export APPTAINERENV_NUMBA_CACHE_DIR="$NUMBA_CACHE_DIR"
export APPTAINERENV_TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export APPTAINERENV_HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET"
export APPTAINERENV_REQUESTS_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt"
export APPTAINERENV_CURL_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt"
export APPTAINERENV_SSL_CERT_FILE="/etc/ssl/certs/ca-certificates.crt"

# Optional: pull once to SIF in scratch, then reuse
SIF_PATH="${SCRATCH:-/tmp/$USER}/vllm-openai-nightly.sif"
apptainer pull "$SIF_PATH" docker://vllm/vllm-openai:nightly

# 3) Verify container can see the GPU
srun -N1 -n1 apptainer exec --cleanenv --nv --bind "$APPTAINER_BINDPATH" --home "$APPTAINER_HOME" "$SIF_PATH" nvidia-smi -L

# 4) Optional: verify CLI flags supported by this image
srun -N1 -n1 apptainer exec --cleanenv --nv --bind "$APPTAINER_BINDPATH" --home "$APPTAINER_HOME" "$SIF_PATH" vllm serve --help | head -n 60

# 5) Launch vLLM from container.
#    Do NOT add "--device cuda" here; many nightly images reject that flag.
srun -N1 -n1 apptainer exec --cleanenv --nv --bind "$APPTAINER_BINDPATH" --home "$APPTAINER_HOME" "$SIF_PATH" \
  vllm serve "$MODEL_ID" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --port "$PORT"
```

If you see `Failed to infer device type` or NVML warnings, you are usually on a login node or outside Slurm GPU cgroup enforcement. Re-run via `srun` inside an active GPU allocation.

## 4) Verify Environment

```bash
python - <<'PY'
import torch, cv2, vllm, openai, pycocotools, timm, ftfy, skimage, sklearn, pandas, matplotlib
print("OK  torch")
print("OK  cv2")
print("OK  vllm")
print("OK  openai")
print("OK  pycocotools")
print("OK  timm")
print("OK  ftfy")
print("OK  skimage")
print("OK  sklearn")
print("OK  pandas")
print("OK  matplotlib")
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
export HF_ASSETS_CACHE="${HF_HOME}/assets"
export TORCH_HOME="${HF_HOME}/torch"
export XDG_CACHE_HOME="${HF_HOME}/xdg"
export TMPDIR="${SLURM_TMPDIR:-${JOB_CACHE_ROOT}/tmp}"
export VLLM_CACHE_ROOT="${JOB_CACHE_ROOT}/vllm"
export TRITON_CACHE_DIR="${JOB_CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${JOB_CACHE_ROOT}/torchinductor"
export NUMBA_CACHE_DIR="${JOB_CACHE_ROOT}/numba"
export HF_HUB_DISABLE_XET=1
unset TRANSFORMERS_CACHE
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$HF_ASSETS_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TMPDIR" \
  "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$NUMBA_CACHE_DIR"
```

One-time cleanup of stale home-cache data (review first, then delete):

```bash
# Review biggest home-cache directories
du -xhd1 ~/.cache ~/.apptainer ~/.singularity 2>/dev/null | sort -h

# Remove common heavy caches from old runs
rm -rf ~/.cache/huggingface ~/.cache/vllm ~/.cache/torch ~/.cache/triton
rm -rf ~/.apptainer/cache ~/.singularity/cache

# Optional: clear stale pip cache in home if present
rm -rf ~/.cache/pip
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
  --temporal_keyframe_pipeline \
  --discovery_mode hybrid \
  --mllm_discovery_window_size 4 \
  --mllm_discovery_window_stride 24 \
  --mllm_discovery_max_json_retries 2 \
  --max_keyframes 6 \
  --min_keyframe_gap 24 \
  --invalid_frame_source mllm \
  --mllm_invalid_window_size 4 \
  --mllm_invalid_window_stride 4 \
  --drop_invalid_frames \
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
- `No module named 'timm'`, `ftfy`, `skimage`, `sklearn`, `pandas`, or `matplotlib` during run:
  - The Nibi runner/sbatch templates now auto-install runtime deps into the active venv using Alliance wheelhouse (`--no-index`) first, then PyPI fallback:
    - `timm>=1.0.17`, `ftfy==6.1.1`, `scikit-image`, `scikit-learn`, `pandas`, `matplotlib`
  - If you still see this, verify the expected venv is being used (`--venv-path`) and that pip install in that venv is writable.
  - Manual repair (current venv): `python -m pip install "timm>=1.0.17" "ftfy==6.1.1" scikit-image scikit-learn pandas matplotlib`
- `numpy.dtype size changed ... pycocotools._mask`:
  - This is an ABI mismatch (typically `numpy 2.x` with a `pycocotools` build expecting `numpy 1.x`).
  - Nibi runner/sbatch templates now auto-repair by force-reinstalling `numpy>=1.26,<2` and `pycocotools` (wheelhouse first, PyPI fallback).
  - Manual repair (current venv): `python -m pip install --force-reinstall "numpy>=1.26,<2" pycocotools`
- `Could not find bpe_simple_vocab_16e6.txt.gz`:
  - Latest script resolves `sam3/assets/...`; optionally set:
    - `export SAM3_BPE_PATH=/home/$USER/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz`
- `World size (2) > available GPUs (1)`:
  - Lower `--tensor-parallel-size` to visible GPU count, or request more GPUs in `salloc`.
- `Disk quota exceeded`:
  - Ensure all HF caches are on `$SLURM_TMPDIR` or `/project`, not `/home`.
- `Read-only file system: .../torchinductor` (Apptainer):
  - This means container cache dirs were not overridden under `--cleanenv`.
  - Set `APPTAINERENV_TORCHINDUCTOR_CACHE_DIR`, `APPTAINERENV_TRITON_CACHE_DIR`, `APPTAINERENV_VLLM_CACHE_ROOT`, and run `apptainer exec` with `--home "$APPTAINER_HOME"` and `--bind "$APPTAINER_BINDPATH"`.
- `--limit-mm-per-prompt` parse error:
  - Use JSON format:
    - `--limit-mm-per-prompt '{"image":1,"video":0}'`
- Context error (`input+output > context length`):
  - Reduce `--max_completion_tokens` (start with 256).
  - Keep `SAM3_IMAGE_DETAIL=low`, `SAM3_AGENT_IMAGE_MAX_EDGE=768`.
  - Qwen3.5 runs now pass `--reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}'` to reduce verbose thinking outputs.
  - Agent client now auto-retries on context overflow by reducing completion budget and (if needed) further downscaling image edge size.
  - Increase `--max-model-len` only after model load is stable.
- Temporal runs should ignore corrupt frames before keyframe propagation:
  - `run_video_agent_openai.py` now supports an MLLM-first frame-validity stage over the full video.
  - Use `--invalid_frame_source mllm` (or default `hybrid`) and tune `--mllm_invalid_window_size/--mllm_invalid_window_stride`.
  - Per-frame valid/invalid output is saved to `frame_quality_scan_mllm.json`.
- `model type qwen3_5 / qwen3_5_moe not recognized`:
  - Your environment is too old for Qwen3.5.
  - Use the separate `.venv-qwen35` flow in section `3.4`.
  - Submit with `--venv-path ~/sam3/.venv-qwen35`.

## 9) Qwen3.5 submit example (batch)

```bash
cd ~/sam3
export ACCOUNT="${ACCOUNT:-rpp-kmoran}"

nibi_model_compare/run_nibi_agent.sh \
  --mode submit \
  --account "$ACCOUNT" \
  --venv-path "$HOME/sam3/.venv-qwen35" \
  --prompt-profile underwater \
  --vllm-runtime apptainer \
  --apptainer-image "${SCRATCH:-/scratch/$USER}/vllm-openai-nightly.sif" \
  --gpus-per-node h100:2 \
  --model-id "Qwen/Qwen3.5-27B" \
  --port 8006 \
  --tp-size 1 \
  --max-model-len 4096 \
  --max-completion-tokens 2048 \
  --video-path "/project/${ACCOUNT}/${USER}/data/onc/chinacreekclipped.mp4" \
  --debug
```

## 10) Notes On Slurm Templates In This Repo

Templates in `nibi_model_compare/slurm` are configured to:
- load `.env` automatically (if present),
- map `HF_TOKEN` into `HUGGINGFACE_HUB_TOKEN` if needed,
- use `$SLURM_TMPDIR`-backed cache by default with optional `/project` prewarm/sync.
- support `VLLM_RUNTIME=auto|venv|apptainer` (auto routes Qwen3.5 models to Apptainer).
- support `SAM3_AGENT_PROMPT_PROFILE` (default `underwater` in Nibi wrappers/templates). Use `general` to revert to base prompts.

## 11) Security

- Do not commit tokens to git.
- If any PAT/token was exposed in logs/chats, rotate it immediately.
