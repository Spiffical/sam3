from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Callable

import cv2
import numpy as np

from frame_quality import analyze_frame_quality


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _default_prompt_path(prompt_profile: str) -> str:
    base = os.path.join(_repo_root(), "sam3", "agent", "system_prompts")
    profile = (prompt_profile or "").strip().lower()
    if profile == "underwater":
        return os.path.join(base, "system_prompt_frame_validity_underwater.txt")
    return os.path.join(base, "system_prompt_frame_validity_general.txt")


def _read_prompt_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def _resolve_system_prompt(
    prompt_profile: str,
    prompt_template_path: str | None,
) -> str:
    if prompt_template_path:
        if not os.path.exists(prompt_template_path):
            raise FileNotFoundError(
                f"Frame-validity prompt template not found: {prompt_template_path}"
            )
        return _read_prompt_file(prompt_template_path)

    candidate = _default_prompt_path(prompt_profile)
    if os.path.exists(candidate):
        return _read_prompt_file(candidate)

    return (
        "Classify frame validity and return STRICT JSON only with schema "
        '{"frames":[{"frame_index":int,"is_valid":bool,"confidence":float,"reason":str}]}.'
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    fenced = re.findall(
        r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL
    )
    for block in fenced:
        block = block.strip()
        if not block:
            continue
        try:
            parsed = json.loads(block)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    start_positions = [idx for idx, ch in enumerate(text) if ch == "{"]
    for start in start_positions:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : idx + 1]
                    try:
                        parsed = json.loads(candidate)
                    except Exception:
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break
    return None


def _rescale_frame_bgr(frame_bgr: np.ndarray, max_edge: int) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_edge:
        return frame_bgr
    scale = float(max_edge) / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _build_collage(
    frame_items: list[tuple[int, np.ndarray]],
    output_path: str,
    *,
    cols: int = 2,
    tile_max_edge: int = 512,
) -> tuple[int, int]:
    assert len(frame_items) > 0
    cols = max(1, int(cols))
    rows = int(math.ceil(len(frame_items) / float(cols)))

    resized = []
    for frame_idx, frame in frame_items:
        tile = _rescale_frame_bgr(frame, max_edge=tile_max_edge)
        tile = tile.copy()
        cv2.putText(
            tile,
            f"f={frame_idx}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            tile,
            f"f={frame_idx}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        resized.append((frame_idx, tile))

    tile_h = max(tile.shape[0] for _, tile in resized)
    tile_w = max(tile.shape[1] for _, tile in resized)
    collage = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

    for i, (_frame_idx, tile) in enumerate(resized):
        r = i // cols
        c = i % cols
        y0 = r * tile_h
        x0 = c * tile_w
        h, w = tile.shape[:2]
        collage[y0 : y0 + h, x0 : x0 + w] = tile

    cv2.imwrite(output_path, collage)
    return rows, cols


def _build_window_series(
    total_frames: int,
    *,
    window_size: int,
    stride: int,
) -> list[list[int]]:
    total_frames = max(0, int(total_frames))
    if total_frames == 0:
        return []
    window_size = max(1, int(window_size))
    stride = max(1, int(stride))

    windows: list[list[int]] = []
    for start in range(0, total_frames, stride):
        end = min(total_frames, start + window_size)
        window = list(range(start, end))
        if window:
            windows.append(window)

    # Ensure tail coverage.
    tail_start = max(0, total_frames - window_size)
    tail = list(range(tail_start, total_frames))
    if tail and (not windows or windows[-1] != tail):
        windows.append(tail)

    return windows


def _read_selected_frames(video_path: str, frame_indices: list[int]) -> dict[int, np.ndarray]:
    target = sorted(set(int(x) for x in frame_indices))
    frame_set = set(target)
    out: dict[int, np.ndarray] = {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video for MLLM frame-validity scan: {video_path}"
        )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in frame_set:
            out[frame_idx] = frame.copy()
            if len(out) == len(target):
                break
        frame_idx += 1
    cap.release()
    return out


def _sanitize_frame_entries(
    parsed: dict[str, Any] | None,
    window: list[int],
) -> list[dict[str, Any]]:
    if not isinstance(parsed, dict):
        return []
    raw_frames = parsed.get("frames", [])
    if not isinstance(raw_frames, list):
        return []

    allowed = set(int(i) for i in window)
    out: list[dict[str, Any]] = []
    for item in raw_frames:
        if not isinstance(item, dict):
            continue
        try:
            frame_idx = int(item.get("frame_index"))
        except Exception:
            continue
        if frame_idx not in allowed:
            continue

        is_valid_raw = item.get("is_valid")
        if isinstance(is_valid_raw, bool):
            is_valid = bool(is_valid_raw)
        elif isinstance(is_valid_raw, (int, float)):
            is_valid = bool(is_valid_raw)
        else:
            continue

        try:
            confidence = float(item.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        reason = str(item.get("reason", "")).strip()
        out.append(
            {
                "frame_index": frame_idx,
                "is_valid": is_valid,
                "is_invalid": not is_valid,
                "confidence": confidence,
                "reason": reason,
            }
        )
    return out


def discover_invalid_frames_with_mllm(
    *,
    video_path: str,
    send_generate_request_fn: Callable[[list[dict[str, Any]]], str | None],
    initial_text_prompt: str,
    total_frames: int,
    output_dir: str,
    window_size: int = 4,
    window_stride: int = 4,
    use_collage: bool = True,
    collage_cols: int = 2,
    collage_tile_max_edge: int = 512,
    prompt_profile: str = "underwater",
    prompt_template_path: str | None = None,
    max_json_retries: int = 2,
    fill_missing_with_heuristic: bool = True,
) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    if int(total_frames) <= 0:
        return {
            "mode": "mllm",
            "total_frames": 0,
            "invalid_frame_indices": [],
            "invalid_frame_ratio": 0.0,
            "per_frame_classification": [],
            "windows": [],
        }

    windows = _build_window_series(
        int(total_frames), window_size=window_size, stride=window_stride
    )
    needed = sorted(set(i for win in windows for i in win))
    cached_frames = _read_selected_frames(video_path, needed)
    system_prompt = _resolve_system_prompt(
        prompt_profile=prompt_profile,
        prompt_template_path=prompt_template_path,
    )

    per_frame_votes: dict[int, list[dict[str, Any]]] = {
        int(i): [] for i in range(int(total_frames))
    }
    window_outputs: list[dict[str, Any]] = []

    for widx, window in enumerate(windows):
        frame_items: list[tuple[int, np.ndarray]] = []
        for idx in window:
            frame = cached_frames.get(int(idx))
            if frame is not None:
                frame_items.append((int(idx), frame))
        if len(frame_items) == 0:
            continue

        user_content: list[dict[str, Any]] = []
        if use_collage:
            collage_path = os.path.join(output_dir, f"window_{widx:04d}.jpg")
            rows, cols = _build_collage(
                frame_items,
                collage_path,
                cols=collage_cols,
                tile_max_edge=collage_tile_max_edge,
            )
            user_content.append({"type": "image", "image": collage_path})
            temporal_hint = (
                "You are given a collage of consecutive frames in chronological order, "
                f"row-major ({rows} rows x {cols} cols)."
            )
        else:
            for idx, frame in frame_items:
                frame_path = os.path.join(output_dir, f"window_{widx:04d}_f_{idx}.jpg")
                cv2.imwrite(frame_path, frame)
                user_content.append({"type": "image", "image": frame_path})
            temporal_hint = (
                "You are given multiple consecutive frames in chronological order "
                "from first to last."
            )

        frame_list = [int(x[0]) for x in frame_items]
        frame_list_text = ", ".join(str(i) for i in frame_list)
        instruction = (
            f"{temporal_hint}\n"
            "Classify each listed frame as valid or invalid for downstream segmentation/tracking.\n"
            "Invalid means unusable (blank/black/white/corrupt/too degraded).\n"
            "Do not omit any listed frame.\n"
            f"Window frame indices: [{frame_list_text}].\n"
            f"Initial user prompt context: '{initial_text_prompt}'.\n"
            "Return STRICT JSON ONLY with this exact schema:\n"
            '{'
            '"frames":[{"frame_index":int,"is_valid":bool,"confidence":float,"reason":str}]'
            '}\n'
            "Include exactly one entry per listed frame index."
        )
        user_content.append({"type": "text", "text": instruction})

        model_text: str | None = None
        parsed: dict[str, Any] | None = None
        request_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        max_attempts = max(1, int(max_json_retries) + 1)
        for attempt in range(max_attempts):
            model_text = send_generate_request_fn(request_messages)
            parsed = _extract_json_object(model_text or "")
            if isinstance(parsed, dict):
                break
            if attempt + 1 < max_attempts:
                request_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Your last reply was not valid JSON. "
                                    "Reply again with STRICT JSON only and no extra text. "
                                    'Use exactly: {"frames":[{"frame_index":int,"is_valid":bool,"confidence":float,"reason":str}]}.'
                                ),
                            }
                        ],
                    }
                )

        safe_entries = _sanitize_frame_entries(parsed, frame_list)
        for entry in safe_entries:
            per_frame_votes[int(entry["frame_index"])].append(entry)

        window_outputs.append(
            {
                "window_index": int(widx),
                "window_frames": frame_list,
                "raw_text": model_text,
                "parsed_frames": safe_entries,
                "json_parse_success": isinstance(parsed, dict),
            }
        )

    per_frame_classification: list[dict[str, Any]] = []
    invalid_frame_indices: list[int] = []

    for frame_idx in range(int(total_frames)):
        votes = per_frame_votes.get(frame_idx, [])
        if votes:
            valid_conf = sum(
                float(v.get("confidence", 0.0)) for v in votes if bool(v.get("is_valid"))
            )
            invalid_conf = sum(
                float(v.get("confidence", 0.0))
                for v in votes
                if not bool(v.get("is_valid"))
            )
            valid_count = sum(1 for v in votes if bool(v.get("is_valid")))
            invalid_count = len(votes) - valid_count

            is_valid = (valid_conf > invalid_conf) or (
                math.isclose(valid_conf, invalid_conf) and valid_count >= invalid_count
            )
            is_invalid = not is_valid

            chosen_votes = [v for v in votes if bool(v.get("is_valid")) == is_valid]
            chosen_votes.sort(
                key=lambda x: float(x.get("confidence", 0.0)), reverse=True
            )
            reason = str(chosen_votes[0].get("reason", "")).strip() if chosen_votes else ""
            conf_denom = max(1e-6, valid_conf + invalid_conf)
            confidence = (
                float(valid_conf if is_valid else invalid_conf) / float(conf_denom)
            )
            source = "mllm_vote"
        else:
            frame = cached_frames.get(frame_idx)
            if frame is None:
                # Should be rare; keep deterministic fallback.
                is_invalid = True
                is_valid = False
                confidence = 0.0
                reason = "missing_frame_data"
                source = "missing"
            elif fill_missing_with_heuristic:
                q = analyze_frame_quality(frame)
                is_invalid = bool(q.get("is_invalid", False))
                is_valid = not is_invalid
                confidence = 0.51
                reason = ",".join(q.get("invalid_reasons", [])) if is_invalid else "heuristic_valid"
                source = "heuristic_fallback"
            else:
                is_invalid = False
                is_valid = True
                confidence = 0.0
                reason = "no_mllm_vote"
                source = "unclassified_default_valid"

        entry = {
            "frame_index": int(frame_idx),
            "is_valid": bool(is_valid),
            "is_invalid": bool(is_invalid),
            "confidence": float(max(0.0, min(1.0, confidence))),
            "reason": str(reason),
            "source": source,
            "num_votes": len(votes),
        }
        per_frame_classification.append(entry)
        if entry["is_invalid"]:
            invalid_frame_indices.append(int(frame_idx))

    invalid_frame_indices = sorted(set(invalid_frame_indices))
    invalid_ratio = (
        len(invalid_frame_indices) / float(total_frames) if total_frames > 0 else 0.0
    )

    return {
        "mode": "mllm",
        "total_frames": int(total_frames),
        "window_size": int(window_size),
        "window_stride": int(window_stride),
        "use_collage": bool(use_collage),
        "collage_cols": int(collage_cols),
        "collage_tile_max_edge": int(collage_tile_max_edge),
        "invalid_frame_indices": invalid_frame_indices,
        "invalid_frame_ratio": float(invalid_ratio),
        "num_invalid_frames": len(invalid_frame_indices),
        "num_windows": len(window_outputs),
        "per_frame_classification": per_frame_classification,
        "windows": window_outputs,
    }
