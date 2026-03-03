#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import cv2
import numpy as np
from pycocotools import mask as mask_util

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from run_video_agent_openai import overlay_masks_on_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an overlay video from frame_outputs_rle.json."
    )
    parser.add_argument("--video_path", required=True, type=str)
    parser.add_argument("--frame_outputs_json", required=True, type=str)
    parser.add_argument("--output_video_path", required=True, type=str)
    parser.add_argument(
        "--drop_invalid_frames",
        action="store_true",
        help="Drop frames listed as invalid in frame_outputs_json metadata.",
    )
    return parser.parse_args()


def decode_rle_to_mask(rle: dict[str, Any]) -> np.ndarray:
    counts = rle.get("counts")
    if isinstance(counts, str):
        rle = dict(rle)
        rle["counts"] = counts.encode("utf-8")
    decoded = mask_util.decode([rle])
    if decoded.ndim == 3:
        return decoded[:, :, 0].astype(bool)
    return decoded.astype(bool)


def load_frame_outputs(path: str) -> tuple[dict[int, dict[str, Any]], list[int]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    frame_map: dict[int, dict[str, Any]] = {}
    invalid_frame_indices = [
        int(idx) for idx in payload.get("invalid_frame_indices", []) if int(idx) >= 0
    ]
    for frame_entry in payload.get("frames", []):
        frame_index = int(frame_entry.get("frame_index", -1))
        if frame_index < 0:
            continue
        rle_masks = frame_entry.get("out_binary_masks_rle", [])
        masks = [decode_rle_to_mask(rle) for rle in rle_masks]
        out_obj_ids = frame_entry.get("out_obj_ids", [])
        out_probs = frame_entry.get("out_probs", [])
        out_boxes_xywh = frame_entry.get("out_boxes_xywh", [])
        frame_map[frame_index] = {
            "out_binary_masks": np.asarray(masks),
            "out_obj_ids": np.asarray(out_obj_ids),
            "out_probs": np.asarray(out_probs),
            "out_boxes_xywh": np.asarray(out_boxes_xywh),
        }
    return frame_map, sorted(set(invalid_frame_indices))


def main() -> int:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_video_path) or ".", exist_ok=True)

    frame_outputs, invalid_frame_indices = load_frame_outputs(args.frame_outputs_json)
    invalid_frame_set = set(invalid_frame_indices)

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        args.output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_w, frame_h),
    )

    frame_index = 0
    dropped = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if args.drop_invalid_frames and frame_index in invalid_frame_set:
            dropped += 1
            frame_index += 1
            continue
        output = frame_outputs.get(frame_index)
        if output is not None:
            frame = overlay_masks_on_frame(frame, output)
        writer.write(frame)
        frame_index += 1

    writer.release()
    cap.release()
    print(
        f"Rendered {frame_index} frames to {args.output_video_path} "
        f"using {len(frame_outputs)} frame outputs; dropped={dropped}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
