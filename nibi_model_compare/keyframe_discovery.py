from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def discover_keyframes_from_motion(
    video_path: str,
    *,
    invalid_frame_indices: set[int] | None = None,
    max_keyframes: int = 6,
    min_keyframe_gap: int = 24,
    motion_threshold: float = 0.03,
) -> dict[str, Any]:
    invalid = invalid_frame_indices or set()
    max_keyframes = max(1, int(max_keyframes))
    min_keyframe_gap = max(1, int(min_keyframe_gap))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for keyframe discovery: {video_path}")

    prev_gray: np.ndarray | None = None
    prev_valid_idx: int | None = None
    first_valid_idx: int | None = None
    motion_scores: list[tuple[int, float, int | None]] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in invalid:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if first_valid_idx is None:
            first_valid_idx = frame_idx

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            score = float(diff.mean() / 255.0)
            motion_scores.append((frame_idx, score, prev_valid_idx))

        prev_gray = gray
        prev_valid_idx = frame_idx
        frame_idx += 1

    cap.release()

    if first_valid_idx is None:
        return {
            "keyframes": [],
            "event_candidates": [],
            "motion_score_count": 0,
        }

    keyframes: list[int] = [first_valid_idx]
    event_candidates: list[dict[str, Any]] = []

    ranked = sorted(motion_scores, key=lambda x: x[1], reverse=True)
    for idx, score, prev_idx in ranked:
        if score < motion_threshold:
            continue
        if idx in invalid:
            continue
        if any(abs(idx - kf) < min_keyframe_gap for kf in keyframes):
            continue
        keyframes.append(idx)
        event_candidates.append(
            {
                "keyframe_idx": int(idx),
                "motion_score": float(score),
                "first_seen_candidate": int(prev_idx if prev_idx is not None else idx),
            }
        )
        if len(keyframes) >= max_keyframes:
            break

    keyframes = sorted(set(keyframes))
    return {
        "keyframes": keyframes,
        "event_candidates": sorted(
            event_candidates, key=lambda x: x["keyframe_idx"]
        ),
        "motion_score_count": len(motion_scores),
    }
