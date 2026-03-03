from __future__ import annotations

import json
import math
import os
import re
from bisect import bisect_left
from typing import Any, Callable

import cv2
import numpy as np


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _default_discovery_prompt_path(prompt_profile: str) -> str:
    base = os.path.join(_repo_root(), "sam3", "agent", "system_prompts")
    profile = (prompt_profile or "").strip().lower()
    if profile == "underwater":
        return os.path.join(base, "system_prompt_temporal_discovery_underwater.txt")
    return os.path.join(base, "system_prompt_temporal_discovery_general.txt")


def _read_prompt_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def _resolve_discovery_system_prompt(
    prompt_profile: str,
    prompt_template_path: str | None,
) -> str:
    if prompt_template_path:
        if not os.path.exists(prompt_template_path):
            raise FileNotFoundError(
                f"Temporal discovery prompt template not found: {prompt_template_path}"
            )
        return _read_prompt_file(prompt_template_path)

    candidate = _default_discovery_prompt_path(prompt_profile)
    if os.path.exists(candidate):
        return _read_prompt_file(candidate)

    # Fallback inline prompt.
    return (
        "You detect NEW object-entry events in ordered frames and respond with STRICT JSON only. "
        'Schema: {"events":[{"first_seen_frame":int,"best_visible_frame":int,"confidence":float,"reason":str}]}. '
        'If none, return {"events":[]}.'
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
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


def _clamp_frame_to_valid(frame_idx: int, valid_frames: list[int]) -> int:
    if not valid_frames:
        return int(frame_idx)
    if frame_idx <= valid_frames[0]:
        return valid_frames[0]
    if frame_idx >= valid_frames[-1]:
        return valid_frames[-1]
    pos = bisect_left(valid_frames, frame_idx)
    if pos >= len(valid_frames):
        return valid_frames[-1]
    if valid_frames[pos] == frame_idx:
        return frame_idx
    prev_idx = valid_frames[pos - 1]
    next_idx = valid_frames[pos]
    return prev_idx if abs(prev_idx - frame_idx) <= abs(next_idx - frame_idx) else next_idx


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


def _read_selected_frames(video_path: str, frame_indices: list[int]) -> dict[int, np.ndarray]:
    target = sorted(set(int(x) for x in frame_indices))
    frame_set = set(target)
    out: dict[int, np.ndarray] = {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for MLLM discovery: {video_path}")

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


def _build_window_series(valid_frame_indices: list[int], window_size: int, stride: int) -> list[list[int]]:
    if len(valid_frame_indices) == 0:
        return []
    window_size = max(2, int(window_size))
    stride = max(1, int(stride))

    windows: list[list[int]] = []
    start_frame = valid_frame_indices[0]
    end_frame = valid_frame_indices[-1]
    for anchor in range(start_frame, end_frame + 1, stride):
        pos = bisect_left(valid_frame_indices, anchor)
        if pos >= len(valid_frame_indices):
            break
        window = valid_frame_indices[pos : pos + window_size]
        if len(window) >= 2:
            windows.append(window)

    # Ensure terminal window is included.
    tail = valid_frame_indices[max(0, len(valid_frame_indices) - window_size) :]
    if len(tail) >= 2:
        windows.append(tail)

    deduped: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for win in windows:
        key = tuple(win)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(win)
    return deduped


def discover_keyframes_with_mllm(
    *,
    video_path: str,
    send_generate_request_fn: Callable[[list[dict[str, Any]]], str | None],
    initial_text_prompt: str,
    valid_frame_indices: list[int],
    output_dir: str,
    max_keyframes: int = 6,
    min_keyframe_gap: int = 24,
    window_size: int = 4,
    window_stride: int = 24,
    min_confidence: float = 0.45,
    max_events: int = 10,
    use_collage: bool = True,
    collage_cols: int = 2,
    collage_tile_max_edge: int = 512,
    prompt_profile: str = "underwater",
    prompt_template_path: str | None = None,
    max_json_retries: int = 2,
) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    valid = sorted(set(int(x) for x in valid_frame_indices))
    if len(valid) == 0:
        return {
            "keyframes": [],
            "event_candidates": [],
            "windows": [],
            "mode": "mllm",
        }

    windows = _build_window_series(valid, window_size=window_size, stride=window_stride)
    all_frame_indices_needed = sorted(set(i for win in windows for i in win))
    cached_frames = _read_selected_frames(video_path, all_frame_indices_needed)
    discovery_system_prompt = _resolve_discovery_system_prompt(
        prompt_profile=prompt_profile,
        prompt_template_path=prompt_template_path,
    )

    window_outputs: list[dict[str, Any]] = []
    raw_events: list[dict[str, Any]] = []

    for widx, window in enumerate(windows):
        frame_items: list[tuple[int, np.ndarray]] = []
        for idx in window:
            frame = cached_frames.get(idx)
            if frame is None:
                continue
            frame_items.append((idx, frame))
        if len(frame_items) < 2:
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

        frame_list_text = ", ".join(str(x[0]) for x in frame_items)
        profile = (prompt_profile or "").strip().lower()
        if profile == "underwater":
            task_line = "Task: detect NEW small underwater creature appearances in this window."
            no_event_line = "If no new creature appears, return: {\"events\":[]}"
        else:
            task_line = "Task: detect NEW target-object appearances in this window."
            no_event_line = "If no new relevant object appears, return: {\"events\":[]}"

        base_instruction = (
            f"{temporal_hint}\n"
            f"{task_line}\n"
            "A new event means a target is absent in earlier frames of this window and appears later.\n"
            "Ignore background mud/sediment changes and ignore blank/corrupt-looking frames.\n"
            f"Window frame indices: [{frame_list_text}].\n"
            f"Initial user prompt context: '{initial_text_prompt}'.\n"
            "Return STRICT JSON ONLY with this schema:\n"
            '{'
            '"events":[{"first_seen_frame":int,"best_visible_frame":int,"confidence":float,"reason":str}]'
            '}\n'
            f"{no_event_line}"
        )
        user_content.append({"type": "text", "text": base_instruction})

        model_text: str | None = None
        parsed: dict[str, Any] | None = None
        request_messages = [
            {"role": "system", "content": discovery_system_prompt},
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
                                    'Use exactly: {"events":[{"first_seen_frame":int,"best_visible_frame":int,"confidence":float,"reason":str}]} '
                                    'or {"events":[]}.'
                                ),
                            }
                        ],
                    }
                )

        parsed_events = parsed.get("events", []) if isinstance(parsed, dict) else []
        safe_events: list[dict[str, Any]] = []
        for evt in parsed_events:
            if not isinstance(evt, dict):
                continue
            fs = evt.get("first_seen_frame")
            bv = evt.get("best_visible_frame")
            conf = evt.get("confidence", 0.0)
            try:
                fs_i = int(fs)
                bv_i = int(bv)
                conf_f = float(conf)
            except Exception:
                continue
            fs_i = _clamp_frame_to_valid(fs_i, valid)
            bv_i = _clamp_frame_to_valid(bv_i, valid)
            if fs_i > bv_i:
                fs_i, bv_i = bv_i, fs_i
            conf_f = max(0.0, min(1.0, conf_f))
            reason = str(evt.get("reason", "")).strip()
            safe_evt = {
                "first_seen_frame": fs_i,
                "best_visible_frame": bv_i,
                "confidence": conf_f,
                "reason": reason,
                "window_index": int(widx),
                "window_frames": [int(i) for i in window],
            }
            safe_events.append(safe_evt)
            raw_events.append(safe_evt)

        window_outputs.append(
            {
                "window_index": int(widx),
                "window_frames": [int(i) for i in window],
                "raw_text": model_text,
                "parsed_events": safe_events,
                "json_parse_success": isinstance(parsed, dict),
            }
        )

    filtered = [
        evt for evt in raw_events if float(evt.get("confidence", 0.0)) >= min_confidence
    ]
    filtered.sort(key=lambda x: (-float(x["confidence"]), int(x["best_visible_frame"])))

    deduped_events: list[dict[str, Any]] = []
    for evt in filtered:
        best_frame = int(evt["best_visible_frame"])
        if any(
            abs(best_frame - int(prev["best_visible_frame"])) < int(min_keyframe_gap)
            for prev in deduped_events
        ):
            continue
        deduped_events.append(evt)
        if len(deduped_events) >= max_events:
            break

    best_visible_frames = sorted(
        set(int(evt["best_visible_frame"]) for evt in deduped_events)
    )
    first_frame = int(valid[0])
    keyframes = [first_frame]
    for idx in best_visible_frames:
        if all(abs(idx - existing) >= int(min_keyframe_gap) for existing in keyframes):
            keyframes.append(int(idx))
        if len(keyframes) >= int(max_keyframes):
            break

    keyframes = sorted(set(keyframes))
    return {
        "mode": "mllm",
        "keyframes": keyframes,
        "event_candidates": sorted(
            deduped_events, key=lambda x: int(x["best_visible_frame"])
        ),
        "windows": window_outputs,
        "num_windows": len(window_outputs),
        "num_events_raw": len(raw_events),
        "num_events_filtered": len(filtered),
        "num_events_selected": len(deduped_events),
    }
