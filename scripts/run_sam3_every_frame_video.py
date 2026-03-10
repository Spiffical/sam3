#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = "small creatures"
DEFAULT_CODEC = "mp4v"
PALETTE = [
    (74, 214, 109),
    (76, 173, 255),
    (255, 198, 71),
    (255, 111, 145),
    (171, 120, 255),
    (120, 235, 232),
]

cv2 = None
np = None
torch = None
Image = None
Sam3Processor = Any
build_sam3_image_model = None


def ensure_runtime_deps() -> None:
    global cv2, np, torch, Image, Sam3Processor, build_sam3_image_model
    if cv2 is not None and np is not None and torch is not None and Image is not None:
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
        torch = importlib.import_module("torch")
    except ImportError:
        missing.append("torch")
    try:
        Image = importlib.import_module("PIL.Image").Image
        from PIL import Image as pil_image_module

        Image = pil_image_module
    except ImportError:
        missing.append("pillow")

    if missing:
        raise RuntimeError(
            "Missing runtime dependencies: "
            + ", ".join(sorted(missing))
            + ". Install them in your local environment first."
        )

    from sam3.model.sam3_image_processor import Sam3Processor as _Sam3Processor
    from sam3.model_builder import (
        build_sam3_image_model as _build_sam3_image_model,
    )

    Sam3Processor = _Sam3Processor
    build_sam3_image_model = _build_sam3_image_model


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


@dataclass
class FrameResult:
    masks: np.ndarray
    boxes_xyxy: np.ndarray
    scores: np.ndarray


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
            or self.current % 10 == 0
        )
        if should_print:
            elapsed = max(1e-6, time.time() - self.start_time)
            fps = self.current / elapsed
            suffix = ""
            if postfix:
                suffix = " | " + ", ".join(f"{k}={v}" for k, v in postfix.items())
            print(
                f"\r{self.desc}: {self.current}/{self.total} "
                f"({fps:.2f} frames/s){suffix}",
                end="" if self.current < self.total else "\n",
                flush=True,
            )

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SAM3 image segmentation on every frame of a video and render an "
            "overlay video."
        )
    )
    parser.add_argument("video_path", help="Input video path.")
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=f"Text prompt to use for every frame. Default: {DEFAULT_PROMPT!r}",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Default: <video_stem>_sam3_every_frame next to the input.",
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
        "--confidence-threshold",
        type=float,
        default=0.40,
        help="SAM3 processor confidence threshold. Default: 0.40",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum prediction score to draw. Default: 0.0",
    )
    parser.add_argument(
        "--min-mask-area",
        type=int,
        default=16,
        help="Minimum mask area in pixels to draw. Default: 16",
    )
    parser.add_argument(
        "--mask-alpha",
        type=float,
        default=0.30,
        help="Mask overlay alpha. Default: 0.30",
    )
    parser.add_argument(
        "--box-thickness",
        type=int,
        default=2,
        help="Bounding box thickness. Default: 2",
    )
    parser.add_argument(
        "--label-font-scale",
        type=float,
        default=0.55,
        help="OpenCV font scale for labels. Default: 0.55",
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
        "--max-frames",
        type=int,
        default=0,
        help="Optional frame limit for quick tests. Default: 0 (all frames).",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable model compilation when building the image model.",
    )
    return parser.parse_args()


def make_output_paths(args: argparse.Namespace) -> tuple[str, str]:
    video_path = Path(args.video_path).resolve()
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = video_path.with_name(f"{video_path.stem}_sam3_every_frame")
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
    return str(output_video_path), str(summary_path)


def run_frame_inference(
    processor: Sam3Processor,
    frame_bgr: np.ndarray,
    prompt: str,
) -> FrameResult:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)

    with torch.inference_mode():
        state = processor.set_image(image)
        state = processor.set_text_prompt(state=state, prompt=prompt)

    masks = state["masks"]
    boxes = state["boxes"]
    scores = state["scores"]

    if torch.is_tensor(masks):
        masks = masks.squeeze(1).detach().cpu().numpy()
    else:
        masks = np.asarray(masks).squeeze(1)
    if torch.is_tensor(boxes):
        boxes = boxes.detach().cpu().numpy()
    else:
        boxes = np.asarray(boxes)
    if torch.is_tensor(scores):
        scores = scores.detach().cpu().numpy()
    else:
        scores = np.asarray(scores)

    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]
    masks = masks.astype(bool)
    boxes = boxes.reshape(-1, 4) if boxes.size else np.zeros((0, 4), dtype=np.float32)
    scores = scores.reshape(-1) if scores.size else np.zeros((0,), dtype=np.float32)

    if scores.size:
        order = np.argsort(-scores)
        masks = masks[order]
        boxes = boxes[order]
        scores = scores[order]

    return FrameResult(masks=masks, boxes_xyxy=boxes, scores=scores)


def render_overlay(
    frame_bgr: np.ndarray,
    frame_result: FrameResult,
    *,
    mask_alpha: float,
    min_score: float,
    min_mask_area: int,
    box_thickness: int,
    label_font_scale: float,
) -> tuple[np.ndarray, int]:
    output = frame_bgr.copy()
    drawn = 0
    alpha = float(max(0.0, min(1.0, mask_alpha)))

    for idx, (mask, box_xyxy, score) in enumerate(
        zip(frame_result.masks, frame_result.boxes_xyxy, frame_result.scores)
    ):
        if float(score) < float(min_score):
            continue
        mask_bool = np.asarray(mask).astype(bool)
        if int(mask_bool.sum()) < int(min_mask_area):
            continue

        color = np.array(PALETTE[idx % len(PALETTE)], dtype=np.float32)
        color_mask = np.zeros_like(output, dtype=np.float32)
        color_mask[mask_bool] = color
        blended = output.astype(np.float32)
        blended[mask_bool] = (
            ((1.0 - alpha) * blended[mask_bool]) + (alpha * color_mask[mask_bool])
        )
        output = blended.astype(np.uint8)

        mask_u8 = (mask_bool.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            output,
            contours,
            -1,
            tuple(int(x) for x in color.tolist()),
            max(1, int(box_thickness)),
        )

        x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy.tolist()]
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            tuple(int(x) for x in color.tolist()),
            max(1, int(box_thickness)),
        )
        label = f"{idx + 1}:{float(score):.2f}"
        cv2.putText(
            output,
            label,
            (x1, max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            float(label_font_scale),
            tuple(int(x) for x in color.tolist()),
            1,
            cv2.LINE_AA,
        )
        drawn += 1

    return output, drawn


def main() -> int:
    args = parse_args()
    ensure_runtime_deps()
    output_video_path, summary_path = make_output_paths(args)

    video_path = str(Path(args.video_path).resolve())
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    log("Building SAM3 image model...")
    bpe_path = find_bpe_path()
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        device=str(args.device),
        checkpoint_path=(args.checkpoint_path or None),
        compile=bool(args.compile),
    )
    processor = Sam3Processor(
        model, confidence_threshold=float(args.confidence_threshold)
    )
    log(f"Model ready on device={args.device}")

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

    fourcc = cv2.VideoWriter_fourcc(*str(args.codec))
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_video_path}")

    log(
        f"Processing {max_frames} frame(s) at {fps:.2f} FPS with prompt={args.prompt!r}"
    )
    progress = ProgressReporter(total=max_frames, desc="SAM3")
    start_time = time.time()
    frame_index = 0
    total_drawn_masks = 0
    frames_with_masks = 0

    try:
        while frame_index < max_frames:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_result = run_frame_inference(processor, frame_bgr, args.prompt)
            overlaid_frame, num_drawn = render_overlay(
                frame_bgr,
                frame_result,
                mask_alpha=args.mask_alpha,
                min_score=args.min_score,
                min_mask_area=args.min_mask_area,
                box_thickness=args.box_thickness,
                label_font_scale=args.label_font_scale,
            )
            writer.write(overlaid_frame)

            total_drawn_masks += int(num_drawn)
            if num_drawn > 0:
                frames_with_masks += 1
            frame_index += 1
            avg_masks = total_drawn_masks / float(max(1, frame_index))
            progress.update(
                1,
                postfix={
                    "drawn": num_drawn,
                    "avg_masks": f"{avg_masks:.2f}",
                },
            )
    finally:
        progress.close()
        cap.release()
        writer.release()

    runtime_sec = time.time() - start_time
    summary = {
        "video_path": video_path,
        "output_video_path": output_video_path,
        "prompt": args.prompt,
        "device": args.device,
        "confidence_threshold": float(args.confidence_threshold),
        "min_score": float(args.min_score),
        "min_mask_area": int(args.min_mask_area),
        "mask_alpha": float(args.mask_alpha),
        "codec": args.codec,
        "fps": float(fps),
        "frame_size_hw": [int(frame_h), int(frame_w)],
        "requested_max_frames": int(max_frames),
        "processed_frames": int(frame_index),
        "frames_with_masks": int(frames_with_masks),
        "total_drawn_masks": int(total_drawn_masks),
        "avg_drawn_masks_per_frame": (
            float(total_drawn_masks) / float(max(1, frame_index))
        ),
        "runtime_sec": float(runtime_sec),
        "throughput_fps": float(frame_index) / float(max(1e-6, runtime_sec)),
        "finished_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    log(f"Wrote overlay video: {output_video_path}")
    log(f"Wrote summary JSON: {summary_path}")
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
