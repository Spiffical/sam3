#!/usr/bin/env python3
from __future__ import annotations
"""
Run SAM3 video agent workflow with any OpenAI-compatible VLM backend.

This script mirrors the Gemini-based workflow in sam3/apps/gemini_video_agent.py
but swaps model calls to sam3.agent.client_llm.send_generate_request.
"""

import argparse
import json
import os
import sys
import time
import traceback
from importlib import resources as importlib_resources
from functools import partial
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None

try:
    from PIL import Image
except ImportError:
    Image = None

# Ensure repo root importability when running from copied folder.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
if load_dotenv is not None:
    load_dotenv(os.path.join(REPO_ROOT, ".env"))


def configure_job_cache_defaults() -> None:
    """Default Hugging Face/PyTorch caches to job-local storage on Slurm."""
    if os.environ.get("SAM3_DISABLE_AUTO_CACHE_SETUP") == "1":
        return

    slurm_tmpdir = os.environ.get("SLURM_TMPDIR")
    if not slurm_tmpdir:
        return

    # Respect explicit user configuration from environment or .env.
    if any(
        os.environ.get(key)
        for key in (
            "HF_HOME",
            "HF_HUB_CACHE",
            "HUGGINGFACE_HUB_CACHE",
            "TRANSFORMERS_CACHE",
            "TORCH_HOME",
            "XDG_CACHE_HOME",
        )
    ):
        return

    cache_root = os.environ.get(
        "SAM3_JOB_CACHE_ROOT", os.path.join(slurm_tmpdir, "hf-cache")
    )
    hf_home = os.path.join(cache_root, "hf")
    hf_hub_cache = os.path.join(hf_home, "hub")
    hf_xet_cache = os.path.join(hf_home, "xet")
    torch_home = os.path.join(hf_home, "torch")
    xdg_cache_home = os.path.join(hf_home, "xdg")
    tmpdir = os.path.join(slurm_tmpdir, "tmp")

    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HF_HUB_CACHE", hf_hub_cache)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hf_hub_cache)
    os.environ.setdefault("HF_XET_CACHE", hf_xet_cache)
    os.environ.setdefault("TORCH_HOME", torch_home)
    os.environ.setdefault("XDG_CACHE_HOME", xdg_cache_home)
    os.environ.setdefault("TMPDIR", tmpdir)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TRANSFORMERS_CACHE", hf_hub_cache)

    for path in (
        os.environ["HF_HOME"],
        os.environ["HF_HUB_CACHE"],
        os.environ["HF_XET_CACHE"],
        os.environ["TORCH_HOME"],
        os.environ["XDG_CACHE_HOME"],
        os.environ["TMPDIR"],
    ):
        os.makedirs(path, exist_ok=True)


configure_job_cache_defaults()


def find_bpe_path() -> str:
    env_path = os.environ.get("SAM3_BPE_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    candidates = [
        os.path.join(REPO_ROOT, "assets/bpe_simple_vocab_16e6.txt.gz"),
        os.path.join(REPO_ROOT, "sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
        "assets/bpe_simple_vocab_16e6.txt.gz",
        "sam3/assets/bpe_simple_vocab_16e6.txt.gz",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    # Fallback to packaged resource path when installed/editable.
    try:
        resource_path = str(
            importlib_resources.files("sam3").joinpath(
                "assets/bpe_simple_vocab_16e6.txt.gz"
            )
        )
        if os.path.exists(resource_path):
            return resource_path
    except Exception:
        pass

    raise FileNotFoundError(
        f"Could not find bpe_simple_vocab_16e6.txt.gz in: {candidates}"
    )


def decode_rle_to_mask(rle: Any, height: int, width: int) -> np.ndarray:
    from pycocotools import mask as mask_util

    if isinstance(rle, str):
        rle = {"counts": rle.encode("utf-8"), "size": [height, width]}
    elif isinstance(rle, dict) and "counts" in rle and isinstance(rle["counts"], str):
        rle = dict(rle)
        rle["counts"] = rle["counts"].encode("utf-8")

    decoded = mask_util.decode([rle])
    if decoded.ndim == 3:
        return decoded[:, :, 0]
    return decoded


def get_center_point(mask: np.ndarray) -> tuple[int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    idx = len(ys) // 2
    return int(xs[idx]), int(ys[idx])


def mask_list_from_outputs(outputs: dict[str, Any]) -> list[np.ndarray]:
    if not isinstance(outputs, dict):
        return []
    masks = outputs.get("pred_masks")
    if masks is None:
        masks = outputs.get("video_res_masks")
    if masks is None:
        # Video propagation path returns post-processed masks here.
        masks = outputs.get("out_binary_masks")
    if masks is None and isinstance(outputs.get("obj_id_to_mask"), dict):
        # Some code paths may expose raw object-id keyed masks.
        masks = list(outputs["obj_id_to_mask"].values())
    if masks is None:
        return []
    if isinstance(masks, dict):
        masks = list(masks.values())
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()
    parsed: list[np.ndarray] = []
    for mask in masks:
        arr = np.asarray(mask)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 3:
            arr = arr[0]
        parsed.append(arr > 0)
    return parsed


def object_color(obj_id: int) -> tuple[int, int, int]:
    # Deterministic BGR color per object id.
    return (
        int((obj_id * 47) % 255),
        int((obj_id * 89 + 37) % 255),
        int((obj_id * 131 + 73) % 255),
    )


def iter_output_masks_with_ids(
    outputs: dict[str, Any], frame_h: int, frame_w: int
) -> list[tuple[int, np.ndarray]]:
    if not isinstance(outputs, dict):
        return []

    if "out_binary_masks" in outputs:
        raw_masks = outputs.get("out_binary_masks")
        raw_ids = outputs.get("out_obj_ids")
        if isinstance(raw_masks, torch.Tensor):
            raw_masks = raw_masks.detach().cpu().numpy()
        if isinstance(raw_ids, torch.Tensor):
            raw_ids = raw_ids.detach().cpu().numpy()
        if raw_masks is None:
            return []

        out: list[tuple[int, np.ndarray]] = []
        for i, raw_mask in enumerate(raw_masks):
            arr = np.asarray(raw_mask)
            while arr.ndim > 2:
                arr = arr[0]
            if arr.shape != (frame_h, frame_w):
                arr = cv2.resize(
                    arr.astype(np.float32),
                    (frame_w, frame_h),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask = arr > 0.5
            if raw_ids is not None and len(raw_ids) > i:
                obj_id = int(raw_ids[i])
            else:
                obj_id = i + 1
            out.append((obj_id, mask))
        return out

    # Fallback for legacy outputs without object ids.
    return [(i + 1, m) for i, m in enumerate(mask_list_from_outputs(outputs))]


def overlay_masks_on_frame(video_frame: np.ndarray, outputs: dict[str, Any]) -> np.ndarray:
    frame_h, frame_w = video_frame.shape[:2]
    masks_with_ids = iter_output_masks_with_ids(outputs, frame_h, frame_w)
    if not masks_with_ids:
        return video_frame

    max_area_ratio = float(os.environ.get("SAM3_OVERLAY_MAX_MASK_AREA_RATIO", "0.95"))
    alpha = float(os.environ.get("SAM3_OVERLAY_ALPHA", "0.35"))

    overlay = np.zeros_like(video_frame)
    drawn_any = False

    for obj_id, mask in masks_with_ids:
        if mask.shape != (frame_h, frame_w):
            continue
        area_ratio = float(mask.mean())
        # Guard against runaway masks that can wash out the whole frame.
        if area_ratio <= 0.0 or area_ratio > max_area_ratio:
            continue

        color = object_color(obj_id)
        overlay[mask] = color

        # Draw crisp contours for readability.
        mask_u8 = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(video_frame, contours, -1, color, 2)
        drawn_any = True

    if not drawn_any:
        return video_frame

    return cv2.addWeighted(video_frame, 1.0, overlay, alpha, 0.0)


def _to_serializable_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, list):
        return value
    return list(value)


def _encode_binary_mask_to_rle(mask: np.ndarray) -> dict[str, Any]:
    from pycocotools import mask as mask_util

    arr = np.asarray(mask)
    while arr.ndim > 2:
        arr = arr[0]
    arr = (arr > 0).astype(np.uint8)
    rle = mask_util.encode(np.asfortranarray(arr))
    counts = rle.get("counts")
    if isinstance(counts, bytes):
        rle["counts"] = counts.decode("utf-8")
    return {"size": list(rle.get("size", arr.shape)), "counts": rle["counts"]}


def serialize_frame_output(frame_index: int, outputs: dict[str, Any]) -> dict[str, Any]:
    masks_with_ids = iter_output_masks_with_ids(
        outputs, frame_h=outputs.get("_frame_h", 0), frame_w=outputs.get("_frame_w", 0)
    )
    # If caller did not provide helper dimensions, derive from first mask.
    if masks_with_ids and (outputs.get("_frame_h", 0) == 0 or outputs.get("_frame_w", 0) == 0):
        sample_mask = masks_with_ids[0][1]
        frame_h, frame_w = int(sample_mask.shape[0]), int(sample_mask.shape[1])
    else:
        frame_h = int(outputs.get("_frame_h", 0))
        frame_w = int(outputs.get("_frame_w", 0))

    out_obj_ids = _to_serializable_list(outputs.get("out_obj_ids"))
    out_probs = _to_serializable_list(outputs.get("out_probs"))
    out_boxes_xywh = _to_serializable_list(outputs.get("out_boxes_xywh"))

    rle_masks: list[dict[str, Any]] = []
    obj_ids: list[int] = []
    for obj_id, mask in masks_with_ids:
        if frame_h > 0 and frame_w > 0 and mask.shape != (frame_h, frame_w):
            mask = cv2.resize(
                mask.astype(np.float32),
                (frame_w, frame_h),
                interpolation=cv2.INTER_NEAREST,
            ) > 0.5
        obj_ids.append(int(obj_id))
        rle_masks.append(_encode_binary_mask_to_rle(mask))

    return {
        "frame_index": int(frame_index),
        "out_obj_ids": out_obj_ids if out_obj_ids else obj_ids,
        "out_probs": out_probs,
        "out_boxes_xywh": out_boxes_xywh,
        "out_binary_masks_rle": rle_masks,
    }


def save_frame_outputs_json(
    output_path: str,
    results_by_frame: dict[int, dict[str, Any]],
    frame_h: int,
    frame_w: int,
) -> None:
    frames_payload: list[dict[str, Any]] = []
    for frame_index in sorted(results_by_frame.keys()):
        frame_outputs = dict(results_by_frame[frame_index])
        frame_outputs["_frame_h"] = frame_h
        frame_outputs["_frame_w"] = frame_w
        frames_payload.append(serialize_frame_output(frame_index, frame_outputs))
    payload = {
        "format_version": 1,
        "frame_size_hw": [int(frame_h), int(frame_w)],
        "num_frames_with_outputs": len(frames_payload),
        "frames": frames_payload,
    }
    write_json(output_path, payload)


def write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_run_documentation(output_dir: str, metrics: dict[str, Any]) -> None:
    """Write compact per-run docs for traceability on shared storage."""
    manifest = dict(metrics)
    manifest["recorded_env"] = {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "hf_home": os.environ.get("HF_HOME", ""),
        "transformers_cache": os.environ.get("TRANSFORMERS_CACHE", ""),
    }
    write_json(os.path.join(output_dir, "manifest.json"), manifest)

    lines = [
        "# Run README",
        "",
        "## What This Run Is",
        "",
        "- SAM3 agent-based segmentation/tracking run on one video.",
        "- This folder is self-documenting and safe to archive.",
        "",
        "## Key Inputs",
        "",
        f"- Model: `{metrics.get('model', '')}`",
        f"- Prompt: `{metrics.get('prompt', '')}`",
        f"- Video: `{metrics.get('video_path', '')}`",
        f"- Server URL: `{metrics.get('server_url', '')}`",
        f"- GPUs: `{metrics.get('gpus', '')}`",
        "",
        "## Key Outputs",
        "",
        f"- Metrics JSON: `run_metrics.json`",
        f"- Manifest JSON: `manifest.json`",
        f"- Overlay video: `{metrics.get('output_video_path', '')}`",
        f"- Generated prompts: `{metrics.get('generated_prompts_path', '')}`",
        "",
        "## Runtime",
        "",
        f"- Status: `{metrics.get('status', '')}`",
        f"- Runtime sec: `{metrics.get('runtime_sec', '')}`",
        f"- Finished UTC: `{metrics.get('finished_at_utc', '')}`",
        "",
        "## Slurm Context",
        "",
        f"- SLURM_JOB_ID: `{os.environ.get('SLURM_JOB_ID', '')}`",
        f"- SLURM_ARRAY_TASK_ID: `{os.environ.get('SLURM_ARRAY_TASK_ID', '')}`",
        f"- SLURM_JOB_NODELIST: `{os.environ.get('SLURM_JOB_NODELIST', '')}`",
        "",
    ]
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, type=str)
    parser.add_argument(
        "--prompt",
        default="Identify and segment any biological creatures.",
        type=str,
    )
    parser.add_argument("--server_url", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument(
        "--prompt-profile",
        default=os.environ.get("SAM3_AGENT_PROMPT_PROFILE", "general"),
        type=str,
        help="Agent system-prompt profile (e.g., general, underwater).",
    )
    parser.add_argument("--output_dir", default="nibi_model_compare/run_out", type=str)
    parser.add_argument("--gpus", default="0", type=str)
    parser.add_argument(
        "--image_size",
        default=1008,
        type=int,
        help="Video predictor processing size. For current SAM3 checkpoint, use 1008.",
    )
    parser.add_argument("--max_completion_tokens", default=8000, type=int)
    parser.add_argument(
        "--save_frame_outputs_json",
        action="store_true",
        help="Save propagated per-frame outputs (obj IDs, boxes, masks as RLE) to JSON.",
    )
    parser.add_argument(
        "--frame_outputs_json_path",
        default="",
        type=str,
        help="Optional explicit path for per-frame outputs JSON.",
    )
    parser.add_argument("--save_prompts", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    os.environ["SAM3_AGENT_PROMPT_PROFILE"] = args.prompt_profile
    os.makedirs(args.output_dir, exist_ok=True)
    start_ts = time.time()
    metrics: dict[str, Any] = {
        "status": "started",
        "video_path": os.path.abspath(args.video_path),
        "prompt": args.prompt,
        "prompt_profile": args.prompt_profile,
        "server_url": args.server_url,
        "model": args.model,
        "gpus": args.gpus,
    }

    backend: Any = None
    cap = None

    try:
        if cv2 is None or np is None or torch is None or Image is None:
            raise RuntimeError(
                "Missing required packages. Install dependencies including "
                "opencv-python, numpy, torch, and pillow in this environment."
            )

        from sam3.agent.agent_core import agent_inference
        from sam3.agent.client_llm import (
            send_generate_request as send_generate_request_orig,
        )
        from sam3.agent.client_sam3 import remove_overlapping_masks, sam3_inference
        from sam3.agent.viz import visualize
        from sam3.apps.interactive_video.backend import PredictorBackend
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        class LocalSam3Service:
            """Local adapter for agent_core tool-calling path."""

            def __init__(self, processor: Any, output_dir: str) -> None:
                self.processor = processor
                self.output_dir = output_dir
                os.makedirs(output_dir, exist_ok=True)

            def call_service(
                self,
                image_path: str,
                text_prompt: str,
                output_folder_path: str | None = None,
            ) -> str:
                if output_folder_path is None:
                    output_folder_path = self.output_dir
                os.makedirs(output_folder_path, exist_ok=True)

                outputs = sam3_inference(self.processor, image_path, text_prompt)
                outputs = remove_overlapping_masks(outputs)

                safe_prompt = text_prompt.replace("/", "_").replace(" ", "_")
                out_name = f"{os.path.basename(image_path)}_{safe_prompt}"
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
                visualize(outputs).save(output_image_path)
                return output_json_path

        bpe_path = find_bpe_path()
        image_model = build_sam3_image_model(bpe_path=bpe_path)
        image_processor = Sam3Processor(image_model, confidence_threshold=0.4)
        local_service = LocalSam3Service(
            image_processor, os.path.join(args.output_dir, "sam_service")
        )

        cap = cv2.VideoCapture(args.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {args.video_path}")

        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Could not read frame 0 from video.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_h, frame_w = frame.shape[:2]
        cap.release()
        cap = None

        frame_0_path = os.path.join(args.output_dir, "frame_0.jpg")
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(frame_0_path)

        api_key = (
            args.api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("VLLM_API_KEY")
            or "DUMMY_API_KEY"
        )
        send_req = partial(
            send_generate_request_orig,
            server_url=args.server_url,
            model=args.model,
            api_key=api_key,
            max_tokens=args.max_completion_tokens,
        )

        history, final_outputs, _ = agent_inference(
            img_path=frame_0_path,
            initial_text_prompt=args.prompt,
            send_generate_request=send_req,
            call_sam_service=local_service.call_service,
            output_dir=os.path.join(args.output_dir, "agent_out"),
            debug=args.debug,
        )

        selected_masks_rle = final_outputs.get("pred_masks", [])
        metrics["agent_history_len"] = len(history)
        metrics["num_agent_masks"] = len(selected_masks_rle)

        generated_prompts: list[dict[str, Any]] = []
        session_id: str | None = None

        if not args.save_prompts and len(selected_masks_rle) > 0:
            gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
            backend = PredictorBackend(gpu_ids=gpu_ids)
            session_id = backend.start_session(
                resource_path=args.video_path,
                image_size=args.image_size,
            )
            metrics["video_session_id"] = session_id

        for index, rle in enumerate(selected_masks_rle):
            try:
                mask = decode_rle_to_mask(rle, frame_h, frame_w)
            except Exception as exc:
                print(f"[warn] failed to decode RLE {index}: {exc}")
                continue
            point = get_center_point(mask)
            if point is None:
                continue

            prompt_data = {
                "frame_idx": 0,
                "obj_id": index + 1,
                "points": [[float(point[0]), float(point[1]), 1]],
                "label": 1,
                "source": "agent_center_point",
            }
            generated_prompts.append(prompt_data)

            if backend is not None and session_id is not None:
                backend.add_point_prompt(
                    session_id=session_id,
                    frame_idx=0,
                    obj_id=index + 1,
                    points=[(float(point[0]), float(point[1]), 1)],
                    # PredictorBackend expects frame_size as (width, height).
                    frame_size=(frame_w, frame_h),
                )

        prompts_path = os.path.join(args.output_dir, "generated_prompts.json")
        write_json(prompts_path, {"prompts": generated_prompts})
        metrics["num_generated_prompts"] = len(generated_prompts)
        metrics["generated_prompts_path"] = prompts_path
        metrics["total_video_frames"] = total_frames

        results_by_frame: dict[int, dict[str, Any]] = {}
        output_video_path = ""
        frame_outputs_json_path = ""

        if args.save_prompts:
            metrics["status"] = "success_prompts_only"
        elif backend is None or session_id is None:
            metrics["status"] = "success_no_propagation"
        else:
            try:
                for output in backend.propagate(
                    {"session_id": session_id, "type": "propagate_in_video"}
                ):
                    frame_index = int(output["frame_index"])
                    results_by_frame[frame_index] = output["outputs"]
            except Exception as exc:
                err_msg = str(exc).strip() or repr(exc) or type(exc).__name__
                raise RuntimeError(
                    f"propagate_in_video failed ({type(exc).__name__}): {err_msg}"
                ) from exc

            out_path = os.path.join(args.output_dir, "output_video.mp4")
            should_save_frame_outputs = args.save_frame_outputs_json or os.environ.get(
                "SAM3_SAVE_FRAME_OUTPUTS_JSON", "1"
            ) == "1"
            if should_save_frame_outputs:
                frame_outputs_json_path = (
                    args.frame_outputs_json_path
                    or os.path.join(args.output_dir, "frame_outputs_rle.json")
                )
                save_frame_outputs_json(
                    frame_outputs_json_path,
                    results_by_frame,
                    frame_h=frame_h,
                    frame_w=frame_w,
                )
                metrics["frame_outputs_json_path"] = frame_outputs_json_path

            cap = cv2.VideoCapture(args.video_path)
            writer = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (frame_w, frame_h),
            )

            frame_index = 0
            while True:
                ret, video_frame = cap.read()
                if not ret:
                    break
                output = results_by_frame.get(frame_index)
                if output:
                    video_frame = overlay_masks_on_frame(video_frame, output)
                writer.write(video_frame)
                frame_index += 1

            writer.release()
            cap.release()
            cap = None

            output_video_path = out_path
            metrics["status"] = "success"
            metrics["frames_with_outputs"] = len(results_by_frame)
            if total_frames > 0:
                metrics["frame_output_fraction"] = len(results_by_frame) / total_frames

        metrics["output_video_path"] = output_video_path
        metrics["output_dir"] = os.path.abspath(args.output_dir)
        return_code = 0

    except Exception as exc:
        metrics["status"] = "failed"
        metrics["error"] = str(exc).strip() or repr(exc) or type(exc).__name__
        metrics["error_type"] = type(exc).__name__
        metrics["traceback"] = traceback.format_exc()
        return_code = 1
        print(f"[error] {metrics['error']}")
        print(metrics["traceback"])

    finally:
        if backend is not None:
            try:
                backend.shutdown()
            except Exception as exc:
                print(f"[warn] backend shutdown failed: {exc}")
        if cap is not None:
            cap.release()
        metrics["runtime_sec"] = round(time.time() - start_ts, 3)
        metrics["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(os.path.join(args.output_dir, "run_metrics.json"), metrics)
        write_run_documentation(args.output_dir, metrics)
        print(json.dumps(metrics, indent=2))

    return return_code


if __name__ == "__main__":
    raise SystemExit(run())
