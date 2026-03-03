from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _normalized_entropy_u8(gray_u8: np.ndarray, bins: int = 64) -> float:
    hist = cv2.calcHist([gray_u8], [0], None, [bins], [0, 256]).flatten()
    total = float(hist.sum())
    if total <= 0.0:
        return 0.0
    prob = hist / total
    prob = prob[prob > 0.0]
    if prob.size == 0:
        return 0.0
    entropy = float(-(prob * np.log2(prob)).sum())
    max_entropy = np.log2(float(bins))
    if max_entropy <= 0.0:
        return 0.0
    return entropy / max_entropy


def analyze_frame_quality(
    frame_bgr: np.ndarray,
    *,
    black_mean_threshold: float = 8.0,
    white_mean_threshold: float = 247.0,
    low_std_threshold: float = 2.5,
    low_entropy_threshold: float = 0.08,
) -> dict[str, Any]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    std = float(gray.std())
    entropy = _normalized_entropy_u8(gray)
    p01 = float(np.percentile(gray, 1))
    p99 = float(np.percentile(gray, 99))
    dynamic_range = p99 - p01
    mostly_black_ratio = float((gray <= 10).mean())
    mostly_white_ratio = float((gray >= 245).mean())

    reasons: list[str] = []
    if mostly_black_ratio >= 0.985:
        reasons.append("mostly_black")
    if mostly_white_ratio >= 0.985:
        reasons.append("mostly_white")

    if mean <= black_mean_threshold and std <= low_std_threshold:
        reasons.append("near_black_uniform")
    if mean >= white_mean_threshold and std <= low_std_threshold:
        reasons.append("near_white_uniform")

    # Percentile-based checks are robust to tiny speckle noise in corrupted frames.
    if p99 <= 24.0:
        reasons.append("near_black_percentile")
    if p01 >= 232.0:
        reasons.append("near_white_percentile")

    # Low-information frame detection for washed-out / near-blank frames.
    if dynamic_range <= 18.0 and entropy <= max(0.20, low_entropy_threshold * 2.5):
        reasons.append("low_dynamic_range_low_entropy")
    if entropy <= low_entropy_threshold and std <= (low_std_threshold * 1.5):
        reasons.append("low_entropy_low_variance")
    if std <= (low_std_threshold * 2.0) and entropy <= max(
        0.16, low_entropy_threshold * 2.0
    ):
        reasons.append("low_std_low_entropy")

    return {
        "gray_mean": mean,
        "gray_std": std,
        "entropy": entropy,
        "gray_p01": p01,
        "gray_p99": p99,
        "gray_dynamic_range": dynamic_range,
        "mostly_black_ratio": mostly_black_ratio,
        "mostly_white_ratio": mostly_white_ratio,
        "is_invalid": len(reasons) > 0,
        "invalid_reasons": sorted(set(reasons)),
    }


def scan_video_frame_quality(
    video_path: str,
    *,
    black_mean_threshold: float = 8.0,
    white_mean_threshold: float = 247.0,
    low_std_threshold: float = 2.5,
    low_entropy_threshold: float = 0.08,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for quality scan: {video_path}")

    per_frame: dict[int, dict[str, Any]] = {}
    invalid_frame_indices: list[int] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        q = analyze_frame_quality(
            frame,
            black_mean_threshold=black_mean_threshold,
            white_mean_threshold=white_mean_threshold,
            low_std_threshold=low_std_threshold,
            low_entropy_threshold=low_entropy_threshold,
        )
        per_frame[frame_idx] = q
        if q["is_invalid"]:
            invalid_frame_indices.append(frame_idx)
        frame_idx += 1

    cap.release()

    invalid_ratio = (len(invalid_frame_indices) / frame_idx) if frame_idx > 0 else 0.0
    return {
        "total_frames_scanned": frame_idx,
        "invalid_frame_indices": invalid_frame_indices,
        "invalid_frame_ratio": invalid_ratio,
        "per_frame_quality": per_frame,
    }
