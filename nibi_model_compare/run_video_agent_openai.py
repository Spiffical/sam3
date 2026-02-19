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
        return []
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()
    parsed: list[np.ndarray] = []
    for mask in masks:
        arr = np.asarray(mask)
        if arr.ndim == 3:
            arr = arr[0]
        parsed.append(arr > 0)
    return parsed


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
    parser.add_argument("--output_dir", default="nibi_model_compare/run_out", type=str)
    parser.add_argument("--gpus", default="0", type=str)
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--max_completion_tokens", default=1024, type=int)
    parser.add_argument("--save_prompts", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    start_ts = time.time()
    metrics: dict[str, Any] = {
        "status": "started",
        "video_path": os.path.abspath(args.video_path),
        "prompt": args.prompt,
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
                    frame_size=(frame_h, frame_w),
                )

        prompts_path = os.path.join(args.output_dir, "generated_prompts.json")
        write_json(prompts_path, {"prompts": generated_prompts})
        metrics["num_generated_prompts"] = len(generated_prompts)
        metrics["generated_prompts_path"] = prompts_path
        metrics["total_video_frames"] = total_frames

        results_by_frame: dict[int, dict[str, Any]] = {}
        output_video_path = ""

        if args.save_prompts:
            metrics["status"] = "success_prompts_only"
        elif backend is None or session_id is None:
            metrics["status"] = "success_no_propagation"
        else:
            for output in backend.propagate(
                {"session_id": session_id, "type": "propagate_in_video"}
            ):
                frame_index = int(output["frame_index"])
                results_by_frame[frame_index] = output["outputs"]

            out_path = os.path.join(args.output_dir, "output_video.mp4")
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
                    overlay = np.zeros_like(video_frame)
                    for obj_idx, mask in enumerate(mask_list_from_outputs(output)):
                        color = (
                            int((obj_idx * 47) % 255),
                            int((obj_idx * 89 + 37) % 255),
                            int((obj_idx * 131 + 73) % 255),
                        )
                        overlay[mask] = color
                    video_frame = cv2.addWeighted(video_frame, 1.0, overlay, 0.5, 0.0)
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
        metrics["error"] = str(exc)
        return_code = 1
        print(f"[error] {exc}")

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
