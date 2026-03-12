#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = "small creatures"
DEFAULT_PROMPT_PROFILE = "underwater"
DEFAULT_CODEC = "mp4v"

cv2 = None
np = None
Image = None
torch = None
Sam3Processor = Any
build_sam3_image_model = None
agent_inference = None
send_generate_request_orig = None
sam3_inference = None
remove_overlapping_masks = None
visualize = None
analyze_frame_quality = None
scan_video_frame_quality = None
discover_invalid_frames_with_mllm = None
encode_binary_mask_to_rle = None


def ensure_runtime_deps() -> None:
    global cv2, np, Image, torch
    global Sam3Processor, build_sam3_image_model
    global agent_inference, send_generate_request_orig
    global sam3_inference, remove_overlapping_masks, visualize
    global analyze_frame_quality, scan_video_frame_quality
    global discover_invalid_frames_with_mllm
    global encode_binary_mask_to_rle
    if cv2 is not None and np is not None and Image is not None and torch is not None:
        return

    missing: list[str] = []
    try:
        cv2 = importlib.import_module("cv2")
    except ImportError:
        missing.append("opencv-python")
    try:
        np = importlib.import_module("numpy")
    except ImportError:
        missing.append("numpy")
    try:
        import torch as torch_mod

        torch = torch_mod
    except ImportError:
        missing.append("torch")
    try:
        from PIL import Image as pil_image_module

        Image = pil_image_module
    except ImportError:
        missing.append("pillow")
    try:
        repo_root_str = str(REPO_ROOT)
        nibi_root_str = str(REPO_ROOT / "nibi_model_compare")
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)
        if nibi_root_str not in sys.path:
            sys.path.insert(0, nibi_root_str)
        from sam3.model.sam3_image_processor import Sam3Processor as _Sam3Processor
        from sam3.model_builder import (
            build_sam3_image_model as _build_sam3_image_model,
        )
        from sam3.agent.agent_core import agent_inference as _agent_inference
        from sam3.agent.client_llm import (
            send_generate_request as _send_generate_request_orig,
        )
        from sam3.agent.client_sam3 import (
            remove_overlapping_masks as _remove_overlapping_masks,
            sam3_inference as _sam3_inference,
        )
        from sam3.agent.viz import visualize as _visualize
        from frame_quality import (
            analyze_frame_quality as _analyze_frame_quality,
            scan_video_frame_quality as _scan_video_frame_quality,
        )
        from frame_quality_mllm import (
            discover_invalid_frames_with_mllm as _discover_invalid_frames_with_mllm,
        )
        from frame_output_utils import (
            encode_binary_mask_to_rle as _encode_binary_mask_to_rle,
        )
    except ImportError as exc:
        missing.append(str(exc))
    else:
        Sam3Processor = _Sam3Processor
        build_sam3_image_model = _build_sam3_image_model
        agent_inference = _agent_inference
        send_generate_request_orig = _send_generate_request_orig
        remove_overlapping_masks = _remove_overlapping_masks
        sam3_inference = _sam3_inference
        visualize = _visualize
        analyze_frame_quality = _analyze_frame_quality
        scan_video_frame_quality = _scan_video_frame_quality
        discover_invalid_frames_with_mllm = _discover_invalid_frames_with_mllm
        encode_binary_mask_to_rle = _encode_binary_mask_to_rle

    if missing:
        raise RuntimeError(
            "Missing runtime dependencies: "
            + ", ".join(missing)
            + ". Activate the SAM3 environment on DRAC before running this script."
        )


def default_device() -> str:
    try:
        import torch as torch_mod

        return "cuda" if torch_mod.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def find_bpe_path() -> str:
    env_path = os.environ.get("SAM3_BPE_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    candidates = [
        REPO_ROOT / "assets" / "bpe_simple_vocab_16e6.txt.gz",
        REPO_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz",
        Path("assets") / "bpe_simple_vocab_16e6.txt.gz",
        Path("sam3") / "assets" / "bpe_simple_vocab_16e6.txt.gz",
    ]
    for path in candidates:
        if path.exists():
            return str(path)

    try:
        import importlib_resources

        resource_path = importlib_resources.files("sam3").joinpath(
            "assets/bpe_simple_vocab_16e6.txt.gz"
        )
        if resource_path.is_file():
            return str(resource_path)
    except Exception:
        pass

    raise FileNotFoundError(
        "Could not find bpe_simple_vocab_16e6.txt.gz. "
        "Set SAM3_BPE_PATH or run from a repo checkout with assets present."
    )


def log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


class ProgressReporter:
    def __init__(self, total: int, desc: str) -> None:
        self.total = max(0, int(total))
        self.desc = desc
        self.start_time = time.time()
        self.current = 0
        self._bar = (
            tqdm(total=self.total, desc=self.desc, unit="frame", dynamic_ncols=True)
            if tqdm is not None
            else None
        )

    def update(self, count: int = 1, *, postfix: dict[str, Any] | None = None) -> None:
        self.current += int(count)
        if self._bar is not None:
            self._bar.update(count)
            if postfix:
                self._bar.set_postfix(postfix, refresh=False)
            return

        should_print = (
            self.current == self.total
            or self.current == 1
            or self.current % 5 == 0
        )
        if should_print:
            elapsed = max(1e-6, time.time() - self.start_time)
            fps = self.current / elapsed
            suffix = ""
            if postfix:
                suffix = " | " + ", ".join(f"{k}={v}" for k, v in postfix.items())
            print(
                f"\r{self.desc}: {self.current}/{self.total} ({fps:.2f} frames/s){suffix}",
                end="" if self.current < self.total else "\n",
                flush=True,
            )

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the SAM3 agent loop on every frame of a video and write an overlay video."
        )
    )
    parser.add_argument("video_path", help="Input video path.")
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=f"Initial user prompt for the agent. Default: {DEFAULT_PROMPT!r}",
    )
    parser.add_argument(
        "--prompt-profile",
        default=DEFAULT_PROMPT_PROFILE,
        help=f"SAM3 agent prompt profile. Default: {DEFAULT_PROMPT_PROFILE!r}",
    )
    parser.add_argument(
        "--server-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
        help="OpenAI-compatible server URL for the MLLM.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "Qwen/Qwen3.5-27B"),
        help="OpenAI-compatible model ID for the MLLM.",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Optional API key. Defaults to OPENAI_API_KEY or VLLM_API_KEY if set.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=1024,
        help="Maximum tokens per MLLM completion. Default: 1024",
    )
    parser.add_argument(
        "--max-generations",
        type=int,
        default=10,
        help="Maximum agent generations per frame. Default: 10",
    )
    parser.add_argument(
        "--image-detail",
        choices=["low", "high"],
        default=os.environ.get("SAM3_IMAGE_DETAIL", "high"),
        help="Multimodal image detail setting for MLLM requests. Default: high",
    )
    parser.add_argument(
        "--max-images-per-request",
        type=int,
        default=int(os.environ.get("SAM3_MAX_IMAGES_PER_REQUEST", "3")),
        help="Maximum number of images to keep in one MLLM request. Default: 3",
    )
    parser.add_argument(
        "--agent-image-max-edge",
        type=int,
        default=int(os.environ.get("SAM3_AGENT_IMAGE_MAX_EDGE", "768")),
        help="Maximum image edge for MLLM requests before downscaling. Default: 768",
    )
    parser.add_argument(
        "--agent-image-min-edge",
        type=int,
        default=int(os.environ.get("SAM3_AGENT_IMAGE_MIN_EDGE", "384")),
        help="Minimum image edge to back off to on context overflow. Default: 384",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Default: <video_stem>_sam3_agent_every_frame",
    )
    parser.add_argument(
        "--output-video-path",
        default="",
        help="Optional explicit output video path. Default: <output-dir>/overlay.mp4",
    )
    parser.add_argument(
        "--summary-path",
        default="",
        help="Optional explicit summary JSON path. Default: <output-dir>/summary.json",
    )
    parser.add_argument(
        "--frame-results-path",
        default="",
        help="Optional JSONL file to write per-frame agent results.",
    )
    parser.add_argument(
        "--frame-outputs-path",
        default="",
        help="Optional JSON file to write per-frame mask outputs. Default: <output-dir>/frame_outputs_rle.json",
    )
    parser.add_argument(
        "--codec",
        default=DEFAULT_CODEC,
        help=f"OpenCV fourcc codec. Default: {DEFAULT_CODEC}",
    )
    parser.add_argument(
        "--device",
        default=default_device(),
        help="Model device. Default: cuda if available, otherwise cpu.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="",
        help="Optional local SAM3 checkpoint path.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.40,
        help="SAM3 image processor confidence threshold. Default: 0.40",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable model compilation when building the image model.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional frame limit for quick tests. Default: 0 (all frames).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Keep full per-frame agent debug outputs.",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Preserve per-frame input images and agent artifact folders.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Write the raw frame on agent failure and continue processing.",
    )
    parser.add_argument(
        "--system-prompt-path",
        default="",
        help="Optional override for the base SAM3 agent system prompt.",
    )
    parser.add_argument(
        "--iterative-system-prompt-path",
        default="",
        help="Optional override for the iterative mask-checking system prompt.",
    )
    parser.add_argument(
        "--skip-invalid-frames",
        action="store_true",
        help=(
            "Run a pre-pass invalid/corrupt-frame scan and skip those frames during "
            "agent analysis."
        ),
    )
    parser.add_argument(
        "--invalid-frame-source",
        choices=["heuristic", "mllm", "hybrid"],
        default="mllm",
        help="Source for corrupt-frame detection when --skip-invalid-frames is set. Default: mllm",
    )
    parser.add_argument(
        "--invalid-frame-black-mean-threshold",
        type=float,
        default=8.0,
        help="Heuristic invalid-frame black mean threshold. Default: 8.0",
    )
    parser.add_argument(
        "--invalid-frame-white-mean-threshold",
        type=float,
        default=247.0,
        help="Heuristic invalid-frame white mean threshold. Default: 247.0",
    )
    parser.add_argument(
        "--invalid-frame-low-std-threshold",
        type=float,
        default=2.5,
        help="Heuristic invalid-frame low-std threshold. Default: 2.5",
    )
    parser.add_argument(
        "--invalid-frame-low-entropy-threshold",
        type=float,
        default=0.08,
        help="Heuristic invalid-frame low-entropy threshold. Default: 0.08",
    )
    parser.add_argument(
        "--mllm-invalid-window-size",
        type=int,
        default=8,
        help="Frame-validity MLLM window size. Default: 8",
    )
    parser.add_argument(
        "--mllm-invalid-window-stride",
        type=int,
        default=8,
        help="Frame-validity MLLM window stride. Default: 8",
    )
    parser.add_argument(
        "--mllm-invalid-max-completion-tokens",
        type=int,
        default=1024,
        help="Max completion tokens for invalid-frame MLLM pass. Default: 1024",
    )
    parser.add_argument(
        "--mllm-invalid-prompt-path",
        default="",
        help="Optional system prompt override for invalid-frame MLLM classification.",
    )
    parser.add_argument(
        "--mllm-invalid-max-json-retries",
        type=int,
        default=2,
        help="Maximum JSON repair retries for invalid-frame MLLM classification. Default: 2",
    )
    parser.add_argument(
        "--mllm-invalid-collage-cols",
        type=int,
        default=2,
        help="Number of collage columns for invalid-frame MLLM pass. Default: 2",
    )
    parser.add_argument(
        "--mllm-invalid-collage-tile-max-edge",
        type=int,
        default=768,
        help="Collage tile max edge for invalid-frame MLLM pass. Default: 768",
    )
    return parser.parse_args()


def discover_invalid_frames_for_agent_loop(
    *,
    args: argparse.Namespace,
    video_path: str,
    total_frames: int,
    output_dir: str,
    send_generate_request_fn: Any,
) -> tuple[set[int], dict[str, Any], str]:
    frame_validity_dir = os.path.join(output_dir, "frame_validity")
    os.makedirs(frame_validity_dir, exist_ok=True)

    heuristic_report: dict[str, Any] | None = None
    mllm_report: dict[str, Any] | None = None

    if args.invalid_frame_source in {"heuristic", "hybrid"}:
        heuristic_report = scan_video_frame_quality(
            video_path,
            black_mean_threshold=float(args.invalid_frame_black_mean_threshold),
            white_mean_threshold=float(args.invalid_frame_white_mean_threshold),
            low_std_threshold=float(args.invalid_frame_low_std_threshold),
            low_entropy_threshold=float(args.invalid_frame_low_entropy_threshold),
        )

    if args.invalid_frame_source in {"mllm", "hybrid"}:
        mllm_send_req = partial(
            send_generate_request_fn,
            max_tokens=int(args.mllm_invalid_max_completion_tokens),
        )
        mllm_report = discover_invalid_frames_with_mllm(
            video_path=video_path,
            send_generate_request_fn=mllm_send_req,
            initial_text_prompt=args.prompt,
            total_frames=int(total_frames),
            output_dir=frame_validity_dir,
            window_size=int(args.mllm_invalid_window_size),
            window_stride=int(args.mllm_invalid_window_stride),
            use_collage=True,
            collage_cols=int(args.mllm_invalid_collage_cols),
            collage_tile_max_edge=int(args.mllm_invalid_collage_tile_max_edge),
            prompt_profile=str(args.prompt_profile),
            prompt_template_path=(
                str(Path(args.mllm_invalid_prompt_path).resolve())
                if args.mllm_invalid_prompt_path
                else None
            ),
            max_json_retries=int(args.mllm_invalid_max_json_retries),
            fill_missing_with_heuristic=True,
        )

    heuristic_invalid = (
        set(int(x) for x in heuristic_report.get("invalid_frame_indices", []))
        if heuristic_report
        else set()
    )
    mllm_invalid = (
        set(int(x) for x in mllm_report.get("invalid_frame_indices", []))
        if mllm_report
        else set()
    )

    if args.invalid_frame_source == "heuristic":
        invalid_frames = heuristic_invalid
    elif args.invalid_frame_source == "mllm":
        invalid_frames = mllm_invalid
    else:
        invalid_frames = heuristic_invalid | mllm_invalid

    report = {
        "enabled": True,
        "mode": str(args.invalid_frame_source),
        "invalid_frame_indices": sorted(invalid_frames),
        "invalid_frame_count": len(invalid_frames),
        "invalid_frame_ratio": (
            len(invalid_frames) / float(max(1, int(total_frames)))
        ),
        "heuristic_invalid_frame_indices": sorted(heuristic_invalid),
        "mllm_invalid_frame_indices": sorted(mllm_invalid),
        "heuristic_report": heuristic_report,
        "mllm_report": mllm_report,
        "config": {
            "black_mean_threshold": float(args.invalid_frame_black_mean_threshold),
            "white_mean_threshold": float(args.invalid_frame_white_mean_threshold),
            "low_std_threshold": float(args.invalid_frame_low_std_threshold),
            "low_entropy_threshold": float(args.invalid_frame_low_entropy_threshold),
            "mllm_window_size": int(args.mllm_invalid_window_size),
            "mllm_window_stride": int(args.mllm_invalid_window_stride),
            "mllm_max_completion_tokens": int(
                args.mllm_invalid_max_completion_tokens
            ),
            "mllm_max_json_retries": int(args.mllm_invalid_max_json_retries),
            "mllm_collage_cols": int(args.mllm_invalid_collage_cols),
            "mllm_collage_tile_max_edge": int(
                args.mllm_invalid_collage_tile_max_edge
            ),
        },
    }
    report_path = os.path.join(frame_validity_dir, "invalid_frame_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return invalid_frames, report, report_path


def make_output_paths(args: argparse.Namespace) -> tuple[str, str, str, str]:
    video_path = Path(args.video_path).resolve()
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = video_path.with_name(f"{video_path.stem}_sam3_agent_every_frame")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_video_path = (
        Path(args.output_video_path).resolve()
        if args.output_video_path
        else output_dir / "overlay.mp4"
    )
    summary_path = (
        Path(args.summary_path).resolve()
        if args.summary_path
        else output_dir / "summary.json"
    )
    frame_results_path = (
        Path(args.frame_results_path).resolve()
        if args.frame_results_path
        else output_dir / "frame_results.jsonl"
    )
    frame_outputs_path = (
        Path(args.frame_outputs_path).resolve()
        if args.frame_outputs_path
        else output_dir / "frame_outputs_rle.json"
    )
    return (
        str(output_video_path),
        str(summary_path),
        str(frame_results_path),
        str(frame_outputs_path),
    )


class LocalSam3Service:
    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def call_service(
        self,
        image_path: str,
        text_prompt: str,
        output_folder_path: str | None = None,
    ) -> str:
        if output_folder_path is None:
            raise ValueError("output_folder_path is required for LocalSam3Service")
        os.makedirs(output_folder_path, exist_ok=True)

        outputs = sam3_inference(self.processor, image_path, text_prompt)
        outputs = remove_overlapping_masks(outputs)

        safe_prompt = text_prompt.replace("/", "_").replace(" ", "_")
        out_name = f"{Path(image_path).stem}_{safe_prompt}"
        output_image_path = os.path.join(output_folder_path, f"{out_name}.png")
        output_json_path = os.path.join(output_folder_path, f"{out_name}.json")

        outputs = {
            "original_image_path": image_path,
            "output_image_path": output_image_path,
            **outputs,
        }

        if "pred_scores" in outputs and outputs["pred_scores"]:
            order = sorted(
                range(len(outputs["pred_scores"])),
                key=lambda i: outputs["pred_scores"][i],
                reverse=True,
            )
            outputs["pred_scores"] = [outputs["pred_scores"][i] for i in order]
            outputs["pred_boxes"] = [outputs["pred_boxes"][i] for i in order]
            outputs["pred_masks"] = [outputs["pred_masks"][i] for i in order]

        valid_masks: list[Any] = []
        valid_boxes: list[Any] = []
        valid_scores: list[Any] = []
        for i, rle in enumerate(outputs.get("pred_masks", [])):
            if len(rle) > 4:
                valid_masks.append(rle)
                valid_boxes.append(outputs["pred_boxes"][i])
                valid_scores.append(outputs["pred_scores"][i])
        outputs["pred_masks"] = valid_masks
        outputs["pred_boxes"] = valid_boxes
        outputs["pred_scores"] = valid_scores

        with open(output_json_path, "w", encoding="utf-8") as handle:
            json.dump(outputs, handle, indent=2)
            handle.write("\n")
        visualize(outputs).save(output_image_path)
        return output_json_path


def history_segment_prompts(history: list[dict[str, Any]]) -> list[str]:
    prompts: list[str] = []
    seen: set[str] = set()
    for message in history:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "text":
                continue
            text = str(item.get("text", ""))
            if "segment_phrase" not in text or "<tool>" not in text:
                continue
            start = text.find("<tool>")
            end = text.find("</tool>")
            if start < 0 or end < 0:
                continue
            payload = text[start + len("<tool>") : end].strip()
            try:
                parsed = json.loads(payload)
            except Exception:
                continue
            if parsed.get("name") != "segment_phrase":
                continue
            params = parsed.get("parameters", {})
            prompt = str(params.get("text_prompt", "")).strip()
            if prompt and prompt not in seen:
                prompts.append(prompt)
                seen.add(prompt)
    return prompts


def pil_to_bgr(image: Any) -> Any:
    rgb = image.convert("RGB")
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)


def write_frame_result(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload) + "\n")
    handle.flush()


def _normalize_pred_mask_rle(
    mask_rle: Any,
    *,
    frame_h: int,
    frame_w: int,
) -> dict[str, Any]:
    if isinstance(mask_rle, dict):
        counts = mask_rle.get("counts", "")
        size = mask_rle.get("size") or [int(frame_h), int(frame_w)]
        if isinstance(counts, bytes):
            counts = counts.decode("utf-8")
        return {"size": [int(size[0]), int(size[1])], "counts": str(counts)}
    if isinstance(mask_rle, str):
        return {"size": [int(frame_h), int(frame_w)], "counts": mask_rle}
    return encode_binary_mask_to_rle(np.asarray(mask_rle) > 0)


def serialize_agent_frame_outputs(
    *,
    frame_index: int,
    frame_h: int,
    frame_w: int,
    final_outputs: dict[str, Any] | None,
) -> dict[str, Any]:
    outputs = dict(final_outputs or {})
    pred_masks = list(outputs.get("pred_masks") or [])
    pred_scores = list(outputs.get("pred_scores") or [])
    pred_boxes = list(outputs.get("pred_boxes") or [])

    rle_masks = [
        _normalize_pred_mask_rle(mask_rle, frame_h=frame_h, frame_w=frame_w)
        for mask_rle in pred_masks
    ]
    obj_ids = list(range(1, len(rle_masks) + 1))

    return {
        "frame_index": int(frame_index),
        "out_obj_ids": obj_ids,
        "out_probs": pred_scores[: len(rle_masks)],
        "out_tracker_probs": [],
        "out_boxes_xywh": pred_boxes[: len(rle_masks)],
        "out_binary_masks_rle": rle_masks,
    }


def main() -> int:
    args = parse_args()
    ensure_runtime_deps()

    output_video_path, summary_path, frame_results_path, frame_outputs_path = (
        make_output_paths(args)
    )
    output_dir = str(Path(output_video_path).resolve().parent)
    video_path = str(Path(args.video_path).resolve())
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    os.environ["SAM3_AGENT_PROMPT_PROFILE"] = str(args.prompt_profile)
    os.environ["SAM3_IMAGE_DETAIL"] = str(args.image_detail)
    os.environ["SAM3_MAX_IMAGES_PER_REQUEST"] = str(
        max(1, int(args.max_images_per_request))
    )
    os.environ["SAM3_AGENT_IMAGE_MAX_EDGE"] = str(
        max(128, int(args.agent_image_max_edge))
    )
    os.environ["SAM3_AGENT_IMAGE_MIN_EDGE"] = str(
        max(128, int(args.agent_image_min_edge))
    )
    if args.system_prompt_path:
        os.environ["SAM3_SYSTEM_PROMPT_PATH"] = str(
            Path(args.system_prompt_path).resolve()
        )
    if args.iterative_system_prompt_path:
        os.environ["SAM3_ITERATIVE_SYSTEM_PROMPT_PATH"] = str(
            Path(args.iterative_system_prompt_path).resolve()
        )

    api_key = (
        args.api_key
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("VLLM_API_KEY")
        or "DUMMY_API_KEY"
    )
    send_req_base = partial(
        send_generate_request_orig,
        server_url=args.server_url,
        model=args.model,
        api_key=api_key,
    )
    send_req = partial(send_req_base, max_tokens=int(args.max_completion_tokens))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    max_frames = int(args.max_frames) if int(args.max_frames) > 0 else total_frames
    if max_frames <= 0:
        raise RuntimeError("Video appears to contain no frames.")
    cap.release()

    invalid_frame_set: set[int] = set()
    invalid_frame_report: dict[str, Any] = {
        "enabled": False,
        "mode": "",
        "invalid_frame_indices": [],
        "invalid_frame_count": 0,
        "invalid_frame_ratio": 0.0,
    }
    invalid_frame_report_path = ""
    if args.skip_invalid_frames:
        log(
            "Running pre-pass invalid/corrupt-frame scan "
            f"with source={args.invalid_frame_source!r}..."
        )
        invalid_frame_set, invalid_frame_report, invalid_frame_report_path = (
            discover_invalid_frames_for_agent_loop(
                args=args,
                video_path=video_path,
                total_frames=max_frames,
                output_dir=output_dir,
                send_generate_request_fn=send_req_base,
            )
        )
        log(
            "Corrupt-frame scan complete. "
            f"Skipping {len(invalid_frame_set)} frame(s) during agent analysis."
        )

    log("Building SAM3 image model for the local tool loop...")
    bpe_path = find_bpe_path()
    image_model = build_sam3_image_model(
        bpe_path=bpe_path,
        device=str(args.device),
        checkpoint_path=(args.checkpoint_path or None),
        compile=bool(args.compile),
    )
    image_processor = Sam3Processor(
        image_model, confidence_threshold=float(args.confidence_threshold)
    )
    local_service = LocalSam3Service(image_processor)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*str(args.codec)),
        fps,
        (frame_w, frame_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_video_path}")

    frame_inputs_dir = os.path.join(output_dir, "frame_inputs")
    frame_artifacts_dir = os.path.join(output_dir, "agent_frames")
    if args.keep_artifacts or args.debug:
        os.makedirs(frame_inputs_dir, exist_ok=True)
        os.makedirs(frame_artifacts_dir, exist_ok=True)

    log(
        "Starting frame-by-frame SAM3 agent loop "
        f"for {max_frames} frame(s) with prompt={args.prompt!r}, "
        f"profile={args.prompt_profile!r}, max_generations={args.max_generations}"
    )

    progress = ProgressReporter(total=max_frames, desc="SAM3 agent")
    start_time = time.time()
    processed_frames = 0
    total_masks = 0
    frames_with_masks = 0
    error_count = 0
    skipped_invalid_frames = 0
    analyzed_frames = 0
    frame_output_rows: list[dict[str, Any]] = []

    with open(frame_results_path, "w", encoding="utf-8") as frame_results_handle:
        try:
            for frame_index in range(max_frames):
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                if frame_index in invalid_frame_set:
                    writer.write(frame_bgr)
                    processed_frames += 1
                    skipped_invalid_frames += 1
                    frame_output_rows.append(
                        serialize_agent_frame_outputs(
                            frame_index=frame_index,
                            frame_h=frame_h,
                            frame_w=frame_w,
                            final_outputs={},
                        )
                    )
                    per_frame_payload = {
                        "frame_index": int(frame_index),
                        "num_masks": 0,
                        "segment_prompts": [],
                        "history_len": 0,
                        "frame_runtime_sec": 0.0,
                        "error": None,
                        "skipped": True,
                        "skip_reason": "invalid_corrupt_frame",
                    }
                    write_frame_result(frame_results_handle, per_frame_payload)
                    avg_masks = total_masks / float(max(1, analyzed_frames))
                    progress.update(
                        1,
                        postfix={
                            "masks": 0,
                            "avg_masks": f"{avg_masks:.2f}",
                            "errors": error_count,
                            "skipped": skipped_invalid_frames,
                        },
                    )
                    continue

                frame_name = f"frame_{frame_index:06d}.jpg"
                if args.keep_artifacts or args.debug:
                    frame_path = os.path.join(frame_inputs_dir, frame_name)
                else:
                    frame_path = os.path.join(output_dir, "_current_frame.jpg")
                if not cv2.imwrite(frame_path, frame_bgr):
                    raise RuntimeError(f"Could not write temporary frame image: {frame_path}")

                frame_agent_dir = os.path.join(
                    frame_artifacts_dir if (args.keep_artifacts or args.debug) else output_dir,
                    f"frame_{frame_index:06d}",
                )
                frame_start = time.time()
                frame_error: str | None = None
                history: list[dict[str, Any]] = []
                final_outputs: dict[str, Any] = {
                    "pred_masks": [],
                    "pred_scores": [],
                }

                try:
                    history, final_outputs, rendered_img = agent_inference(
                        img_path=frame_path,
                        initial_text_prompt=args.prompt,
                        debug=bool(args.debug),
                        send_generate_request=send_req,
                        call_sam_service=local_service.call_service,
                        max_generations=int(args.max_generations),
                        output_dir=frame_agent_dir,
                    )
                    overlaid_frame = pil_to_bgr(rendered_img)
                except Exception as exc:
                    frame_error = f"{type(exc).__name__}: {str(exc).strip() or repr(exc)}"
                    error_count += 1
                    if not args.continue_on_error:
                        raise
                    overlaid_frame = frame_bgr.copy()
                finally:
                    if not (args.keep_artifacts or args.debug):
                        try:
                            if os.path.isfile(frame_path):
                                os.remove(frame_path)
                        except Exception:
                            pass
                        try:
                            if os.path.isdir(frame_agent_dir):
                                shutil.rmtree(frame_agent_dir)
                        except Exception:
                            pass

                writer.write(overlaid_frame)

                num_masks = len(final_outputs.get("pred_masks", []))
                segment_prompts = history_segment_prompts(history)
                processed_frames += 1
                analyzed_frames += 1
                total_masks += int(num_masks)
                if num_masks > 0:
                    frames_with_masks += 1

                per_frame_payload = {
                    "frame_index": int(frame_index),
                    "num_masks": int(num_masks),
                    "segment_prompts": segment_prompts,
                    "history_len": int(len(history)),
                    "frame_runtime_sec": float(time.time() - frame_start),
                    "error": frame_error,
                    "skipped": False,
                    "skip_reason": "",
                }
                write_frame_result(frame_results_handle, per_frame_payload)
                frame_output_rows.append(
                    serialize_agent_frame_outputs(
                        frame_index=frame_index,
                        frame_h=frame_h,
                        frame_w=frame_w,
                        final_outputs=final_outputs,
                    )
                )

                avg_masks = total_masks / float(max(1, analyzed_frames))
                progress.update(
                    1,
                    postfix={
                        "masks": num_masks,
                        "avg_masks": f"{avg_masks:.2f}",
                        "errors": error_count,
                        "skipped": skipped_invalid_frames,
                    },
                )
        finally:
            progress.close()
            cap.release()
            writer.release()

    runtime_sec = time.time() - start_time
    summary = {
        "video_path": video_path,
        "output_dir": output_dir,
        "output_video_path": output_video_path,
        "summary_path": summary_path,
        "frame_results_path": frame_results_path,
        "frame_outputs_path": frame_outputs_path,
        "frame_inputs_dir": frame_inputs_dir if (args.keep_artifacts or args.debug) else "",
        "frame_artifacts_dir": (
            frame_artifacts_dir if (args.keep_artifacts or args.debug) else ""
        ),
        "prompt": args.prompt,
        "prompt_profile": args.prompt_profile,
        "server_url": args.server_url,
        "model": args.model,
        "device": args.device,
        "max_generations": int(args.max_generations),
        "max_completion_tokens": int(args.max_completion_tokens),
        "image_detail": str(args.image_detail),
        "max_images_per_request": int(args.max_images_per_request),
        "agent_image_max_edge": int(args.agent_image_max_edge),
        "agent_image_min_edge": int(args.agent_image_min_edge),
        "fps": float(fps),
        "frame_size_hw": [int(frame_h), int(frame_w)],
        "processed_frames": int(processed_frames),
        "analyzed_frames": int(analyzed_frames),
        "frames_with_masks": int(frames_with_masks),
        "total_masks": int(total_masks),
        "avg_masks_per_frame": float(total_masks) / float(max(1, analyzed_frames)),
        "avg_masks_per_analyzed_frame": float(total_masks)
        / float(max(1, analyzed_frames)),
        "error_count": int(error_count),
        "skip_invalid_frames": bool(args.skip_invalid_frames),
        "skipped_invalid_frame_count": int(skipped_invalid_frames),
        "invalid_frame_source": str(args.invalid_frame_source),
        "invalid_frame_report_path": invalid_frame_report_path,
        "invalid_frame_count": int(invalid_frame_report.get("invalid_frame_count", 0)),
        "invalid_frame_ratio": float(
            invalid_frame_report.get("invalid_frame_ratio", 0.0)
        ),
        "invalid_frame_sample": list(
            invalid_frame_report.get("invalid_frame_indices", [])[:20]
        ),
        "keep_artifacts": bool(args.keep_artifacts),
        "debug": bool(args.debug),
        "continue_on_error": bool(args.continue_on_error),
        "runtime_sec": float(runtime_sec),
        "throughput_fps": float(processed_frames) / float(max(1e-6, runtime_sec)),
        "finished_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    frame_outputs_payload = {
        "format_version": 2,
        "source": "sam3_agent_every_frame",
        "frame_size_hw": [int(frame_h), int(frame_w)],
        "total_video_frames": int(processed_frames),
        "num_frames_with_outputs": len(frame_output_rows),
        "invalid_frame_indices": sorted(int(x) for x in invalid_frame_set),
        "keyframe_indices": [],
        "frames": frame_output_rows,
    }
    with open(frame_outputs_path, "w", encoding="utf-8") as handle:
        json.dump(frame_outputs_payload, handle, indent=2)
        handle.write("\n")

    log(f"Wrote overlay video: {output_video_path}")
    log(f"Wrote summary JSON: {summary_path}")
    log(f"Wrote per-frame results: {frame_results_path}")
    log(f"Wrote frame outputs JSON: {frame_outputs_path}")
    if args.keep_artifacts or args.debug:
        log(f"Preserved frame artifacts under: {frame_artifacts_dir}")
    log(
        "Done. Note: the output video is re-encoded via OpenCV and does not preserve audio."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("Interrupted.")
        raise SystemExit(130)
