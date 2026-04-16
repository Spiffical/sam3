from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any


def _empty_frame_row(frame_index: int) -> dict[str, Any]:
    return {
        "frame_index": int(frame_index),
        "out_obj_ids": [],
        "out_probs": [],
        "out_tracker_probs": [],
        "out_boxes_xywh": [],
        "out_binary_masks_rle": [],
    }


def _draw_tile_label(helpers: Any, frame_bgr: Any, text: str) -> Any:
    output = frame_bgr.copy()
    helpers.cv2.putText(
        output,
        text,
        (12, 28),
        helpers.cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        helpers.cv2.LINE_AA,
    )
    helpers.cv2.putText(
        output,
        text,
        (12, 28),
        helpers.cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        1,
        helpers.cv2.LINE_AA,
    )
    return output


def _comparison_tile(
    *,
    helpers: Any,
    frame_bgr: Any,
    frame_index: int,
    mask_items: list[dict[str, Any]],
) -> Any:
    raw = _draw_tile_label(helpers, frame_bgr, f"raw f={int(frame_index)}")
    overlay = _draw_tile_label(
        helpers,
        helpers.draw_overlay_with_labels(frame_bgr, mask_items, assigned_global_ids={}),
        f"overlay f={int(frame_index)}",
    )
    h = max(raw.shape[0], overlay.shape[0])
    w = raw.shape[1] + overlay.shape[1] + 8
    tile = helpers.np.zeros((h, w, 3), dtype=helpers.np.uint8)
    tile[: raw.shape[0], : raw.shape[1]] = raw
    tile[: overlay.shape[0], raw.shape[1] + 8 : raw.shape[1] + 8 + overlay.shape[1]] = overlay
    return tile


def _write_detection_request_images(
    *,
    helpers: Any,
    video_path: str,
    working_frame_rows_by_index: dict[int, dict[str, Any]],
    frame_indices: list[int],
    frame_h: int,
    frame_w: int,
    output_dir: Path,
    max_images_per_request: int,
    collage_cols: int,
    collage_tile_max_edge: int,
) -> tuple[list[str], list[list[int]], list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_frame_tiles: list[tuple[int, Any, str]] = []
    debug_paths: list[str] = []

    for frame_index in frame_indices:
        frame_bgr = helpers.read_video_frame(video_path, frame_index)
        if frame_bgr is None:
            continue
        frame_row = working_frame_rows_by_index.get(frame_index, _empty_frame_row(frame_index))
        mask_items = helpers.decode_frame_row_masks(frame_row, frame_h=frame_h, frame_w=frame_w)
        tile = _comparison_tile(
            helpers=helpers,
            frame_bgr=frame_bgr,
            frame_index=frame_index,
            mask_items=mask_items,
        )
        tile_path = output_dir / f"frame_compare_{int(frame_index):06d}.jpg"
        helpers.cv2.imwrite(str(tile_path), tile)
        per_frame_tiles.append((int(frame_index), tile, str(tile_path)))
        debug_paths.append(str(tile_path))

    if not per_frame_tiles:
        return [], [], []

    image_budget = max(1, int(max_images_per_request))
    if len(per_frame_tiles) <= image_budget:
        return (
            [row[2] for row in per_frame_tiles],
            [[int(row[0])] for row in per_frame_tiles],
            debug_paths,
        )

    request_image_paths: list[str] = []
    request_frame_groups: list[list[int]] = []
    group_count = min(image_budget, len(per_frame_tiles))
    group_size = int(math.ceil(len(per_frame_tiles) / float(group_count)))
    for group_index in range(group_count):
        chunk = per_frame_tiles[group_index * group_size : (group_index + 1) * group_size]
        if not chunk:
            continue
        collage_path = output_dir / f"detection_chunk_{group_index:02d}.jpg"
        helpers.build_collage(
            [(frame_index, tile) for frame_index, tile, _ in chunk],
            str(collage_path),
            cols=min(int(collage_cols), max(1, len(chunk))),
            tile_max_edge=int(collage_tile_max_edge),
        )
        request_image_paths.append(str(collage_path))
        request_frame_groups.append([int(frame_index) for frame_index, _, _ in chunk])
        debug_paths.append(str(collage_path))

    return request_image_paths, request_frame_groups, debug_paths


def _sanitize_detection_response(
    *,
    helpers: Any,
    parsed: dict[str, Any] | None,
    allowed_frame_indices: set[int],
    frame_w: int,
    frame_h: int,
    max_issues: int,
) -> list[dict[str, Any]]:
    if not isinstance(parsed, dict):
        return []
    raw_issues = parsed.get("issues", [])
    if not isinstance(raw_issues, list):
        return []

    issues: list[dict[str, Any]] = []
    seen: set[tuple[int, tuple[tuple[int, int], ...]]] = set()
    for raw_issue in raw_issues:
        if not isinstance(raw_issue, dict):
            continue
        try:
            target_frame_index = int(raw_issue.get("target_frame_index"))
        except Exception:
            continue
        if target_frame_index not in allowed_frame_indices:
            continue

        raw_click_points = raw_issue.get("click_points")
        if raw_click_points is None:
            raw_click_points = raw_issue.get("clicks")
        if raw_click_points is None and isinstance(raw_issue.get("click_point"), (list, tuple)):
            raw_click_points = [raw_issue.get("click_point")]
        if raw_click_points is None and isinstance(raw_issue.get("point"), (list, tuple)):
            raw_click_points = [raw_issue.get("point")]
        if not isinstance(raw_click_points, list):
            continue

        click_points = helpers.dedupe_points(
            [
                point
                for point in (
                    helpers.normalize_point_xy(raw_point, frame_w=frame_w, frame_h=frame_h)
                    for raw_point in raw_click_points
                )
                if point is not None
            ],
            min_distance=4.0,
        )
        if not click_points:
            continue

        try:
            confidence = float(raw_issue.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        description = str(raw_issue.get("description", "")).strip()
        issue_key = (
            int(target_frame_index),
            tuple(sorted((int(point[0]), int(point[1])) for point in click_points)),
        )
        if issue_key in seen:
            continue
        seen.add(issue_key)
        issues.append(
            {
                "target_frame_index": int(target_frame_index),
                "description": description,
                "confidence": confidence,
                "click_points": [[int(point[0]), int(point[1])] for point in click_points[:3]],
            }
        )
        if len(issues) >= max(1, int(max_issues)):
            break
    return issues


def _request_detection_issues(
    *,
    helpers: Any,
    send_generate_request_fn: Any,
    system_prompt: str,
    image_paths: list[str],
    frame_groups: list[list[int]],
    window_frame_indices: list[int],
    max_issues: int,
    max_json_retries: int,
) -> tuple[dict[str, Any] | None, str | None]:
    image_descriptions = "\n".join(
        f"- image {index + 1}: frames {group}"
        for index, group in enumerate(frame_groups)
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                [{"type": "image", "image": image_path} for image_path in image_paths]
                + [
                    {
                        "type": "text",
                        "text": (
                            f"Chronological window frames: {window_frame_indices}\n"
                            f"{image_descriptions}\n\n"
                            "Each image shows one or more frame tiles. Inside each tile, the left half is the raw frame and the right half is the current mask overlay. "
                            "A creature counts as missed only if it is clearly visible in the raw half and not already covered by any mask in the overlay half.\n\n"
                            f"Find at most {int(max_issues)} clear missed-creature cases in this window. "
                            "For each case, choose the single best frame for clicking and provide 1 to 3 positive click points [x,y] in original-frame pixel coordinates. "
                            "Do not report duplicates, already-segmented animals, debris, marine snow, shadows, or ambiguous blobs. "
                            "Return strict JSON only."
                        ),
                    }
                ]
            ),
        },
    ]

    last_text: str | None = None
    for attempt in range(max(1, int(max_json_retries) + 1)):
        last_text = send_generate_request_fn(messages)
        parsed = helpers.extract_json_object(last_text or "")
        if isinstance(parsed, dict):
            return parsed, last_text
        if attempt + 1 < max(1, int(max_json_retries) + 1):
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Your previous reply did not contain valid JSON. "
                                'Reply again with strict JSON only using schema {"issues":[{"target_frame_index":int,"description":str,"click_points":[[x,y]],"confidence":float}]}.'
                            ),
                        }
                    ],
                }
            )
    return None, last_text


def _build_attempt_history_text(attempts: list[dict[str, Any]]) -> str:
    if not attempts:
        return "No previous attempts."
    lines: list[str] = []
    for attempt in attempts[-4:]:
        lines.append(
            f"attempt {int(attempt.get('attempt_index', 0))}: "
            f"points={attempt.get('positive_points', [])}, "
            f"decision={attempt.get('decision', 'unknown')}, "
            f"reason={attempt.get('reason', '')}"
        )
    return "\n".join(lines)


def _build_verification_collage(
    *,
    helpers: Any,
    video_path: str,
    target_frame_index: int,
    target_frame_bgr: Any,
    working_frame_rows_by_index: dict[int, dict[str, Any]],
    current_items: list[dict[str, Any]],
    positive_points: list[tuple[int, int]],
    candidate_item: dict[str, Any] | None,
    output_path: str,
) -> str:
    prompt_overlay = helpers.render_point_prompt_debug(
        target_frame_bgr,
        prompt_points=[(int(x), int(y), 1) for x, y in positive_points],
        hint_bbox_xyxy=None,
        existing_items=current_items,
    )
    candidate_overlay = helpers.draw_mask_focus(
        target_frame_bgr,
        focus_items=[candidate_item] if candidate_item is not None else [],
        existing_items=current_items,
        focus_label_prefix="candidate ",
    )
    current_overlay = helpers.draw_overlay_with_labels(
        target_frame_bgr,
        current_items,
        assigned_global_ids={},
    )

    tiles: list[tuple[int, Any]] = [
        (int(target_frame_index), _draw_tile_label(helpers, target_frame_bgr, "raw target")),
        (int(target_frame_index), _draw_tile_label(helpers, current_overlay, "current overlay")),
        (int(target_frame_index), _draw_tile_label(helpers, prompt_overlay, "prompt overlay")),
        (int(target_frame_index), _draw_tile_label(helpers, candidate_overlay, "candidate overlay")),
    ]

    for ctx_index in range(max(0, int(target_frame_index) - 2), int(target_frame_index) + 3):
        if ctx_index == int(target_frame_index):
            continue
        ctx_frame = helpers.read_video_frame(video_path, ctx_index)
        if ctx_frame is None:
            continue
        ctx_row = working_frame_rows_by_index.get(ctx_index, _empty_frame_row(ctx_index))
        ctx_items = helpers.decode_frame_row_masks(ctx_row, frame_h=ctx_frame.shape[0], frame_w=ctx_frame.shape[1])
        tiles.append(
            (
                int(ctx_index),
                _comparison_tile(
                    helpers=helpers,
                    frame_bgr=ctx_frame,
                    frame_index=ctx_index,
                    mask_items=ctx_items,
                ),
            )
        )

    helpers.build_collage(
        tiles,
        output_path,
        cols=3,
        tile_max_edge=360,
    )
    return output_path


def _request_candidate_verdict(
    *,
    helpers: Any,
    send_generate_request_fn: Any,
    system_prompt: str,
    collage_path: str,
    target_frame_index: int,
    issue_description: str,
    positive_points: list[tuple[int, int]],
    max_existing_iou: float,
    attempt_history_text: str,
    candidate_present: bool,
    max_json_retries: int,
    frame_w: int,
    frame_h: int,
) -> tuple[dict[str, Any], str | None]:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": collage_path},
                {
                    "type": "text",
                    "text": (
                        f"Target frame: {int(target_frame_index)}\n"
                        f"Current positive clicks: {[list(point) for point in positive_points]}\n"
                        f"Issue description: {issue_description or '(none)'}\n"
                        f"Max IoU with any already-present mask on the target frame: {float(max_existing_iou):.3f}\n"
                        f"Candidate mask present: {bool(candidate_present)}\n\n"
                        "The collage includes the raw target frame, the current overlay, the click prompts, the candidate mask, and nearby temporal context. "
                        "Accept only if the candidate clearly adds a real missed creature that is not already segmented. "
                        "If the creature is visible but the candidate is incomplete or wrong, return retry and provide 1 or 2 additional positive click points in original-frame pixel coordinates. "
                        "Keep all existing clicks; any points you return are extra clicks to add, not replacements.\n\n"
                        f"Previous attempts:\n{attempt_history_text}\n\n"
                        "Return strict JSON only."
                    ),
                },
            ],
        },
    ]

    last_text: str | None = None
    parsed: dict[str, Any] | None = None
    for attempt in range(max(1, int(max_json_retries) + 1)):
        last_text = send_generate_request_fn(messages)
        parsed = helpers.extract_json_object(last_text or "")
        if isinstance(parsed, dict):
            break
        if attempt + 1 < max(1, int(max_json_retries) + 1):
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Your previous reply did not contain valid JSON. "
                                'Reply again with strict JSON only using schema {"decision":"accept|retry|reject","reason":str,"additional_points":[[x,y]],"confidence":float}.'
                            ),
                        }
                    ],
                }
            )

    decision = "reject"
    reason = ""
    confidence = 0.0
    additional_points: list[list[int]] = []
    if isinstance(parsed, dict):
        raw_decision = str(parsed.get("decision", "")).strip().lower()
        if raw_decision in {"accept", "retry", "reject"}:
            decision = raw_decision
        reason = str(parsed.get("reason", "")).strip()
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        raw_points = parsed.get("additional_points")
        if raw_points is None and isinstance(parsed.get("suggested_point"), (list, tuple)):
            raw_points = [parsed.get("suggested_point")]
        if raw_points is None and isinstance(parsed.get("suggested_points"), list):
            raw_points = parsed.get("suggested_points")
        if isinstance(raw_points, list):
            additional_points = [
                [int(point[0]), int(point[1])]
                for point in helpers.dedupe_points(
                    [
                        point
                        for point in (
                            helpers.normalize_point_xy(raw_point, frame_w=frame_w, frame_h=frame_h)
                            for raw_point in raw_points
                        )
                        if point is not None
                    ],
                    min_distance=4.0,
                )[:2]
            ]

    return (
        {
            "decision": decision,
            "reason": reason,
            "confidence": confidence,
            "additional_points": additional_points,
        },
        last_text,
    )


def _append_propagated_masks(
    *,
    helpers: Any,
    backend: Any,
    session_id: str,
    frame_h: int,
    frame_w: int,
    working_frame_rows_by_index: dict[int, dict[str, Any]],
    target_frame_index: int,
    accepted_mask_item: dict[str, Any],
    invalid_frame_indices: set[int],
    duplicate_iou_threshold: float,
) -> dict[str, Any]:
    backend.reset_session(session_id)
    backend.add_mask_prompt(
        session_id=session_id,
        frame_idx=int(target_frame_index),
        obj_id=1,
        mask=helpers.np.asarray(accepted_mask_item["mask"]).astype(bool),
    )
    added_by_frame: dict[int, int] = {}
    skipped_duplicates: dict[int, float] = {}
    for output in backend.propagate(
        {
            "session_id": session_id,
            "type": "propagate_in_video",
            "start_frame_index": int(target_frame_index),
            "propagation_direction": "both",
        }
    ):
        frame_index = int(output.get("frame_index", -1))
        if frame_index < 0 or frame_index in invalid_frame_indices:
            continue
        outputs = output.get("outputs") or output
        candidate_item = helpers.select_point_prompt_candidate(
            frame_index=frame_index,
            outputs=outputs,
            requested_obj_id=1,
            frame_h=frame_h,
            frame_w=frame_w,
        )
        if candidate_item is None:
            continue
        frame_row = working_frame_rows_by_index.setdefault(
            int(frame_index),
            _empty_frame_row(frame_index),
        )
        existing_items = helpers.decode_frame_row_masks(frame_row, frame_h=frame_h, frame_w=frame_w)
        max_existing_iou = max(
            (helpers.binary_mask_iou(candidate_item["mask"], existing_item["mask"]) for existing_item in existing_items),
            default=0.0,
        )
        if max_existing_iou >= float(duplicate_iou_threshold):
            skipped_duplicates[int(frame_index)] = float(max_existing_iou)
            continue
        added_local_id = helpers.append_mask_to_frame_row(
            frame_row,
            frame_h=frame_h,
            frame_w=frame_w,
            mask_item=candidate_item,
        )
        added_by_frame[int(frame_index)] = int(added_local_id)
    return {
        "added_local_ids_by_frame": {str(k): int(v) for k, v in sorted(added_by_frame.items())},
        "skipped_duplicate_frames": {str(k): float(v) for k, v in sorted(skipped_duplicates.items())},
    }


def discover_missed_creatures_stage(
    *,
    args: Any,
    send_generate_request_fn: Any,
    detection_system_prompt: str,
    verify_system_prompt: str,
    backend: Any,
    session_id: str,
    video_path: str,
    frame_h: int,
    frame_w: int,
    valid_frame_indices: list[int],
    invalid_frame_indices: set[int],
    working_frame_rows_by_index: dict[int, dict[str, Any]],
    output_dir: Path,
    max_json_retries: int,
    helpers: Any,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "enabled": True,
        "window_count": 0,
        "rounds_completed": 0,
        "detection_requests": [],
        "issues": [],
        "accepted_issue_count": 0,
        "unresolved_issue_count": 0,
        "failure_reason_counts": {},
        "verification_status_counts": {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    windows = helpers.build_index_windows(
        valid_frame_indices,
        window_size=int(args.missed_creatures_window_size),
        stride=int(args.missed_creatures_window_stride),
    )
    report["window_count"] = len(windows)

    for round_index in range(1, max(1, int(args.missed_creatures_max_rounds)) + 1):
        accepted_this_round = 0
        report["rounds_completed"] = int(round_index)
        for window_index, window_frame_indices in enumerate(windows):
            request_dir = output_dir / f"round_{round_index:02d}" / f"window_{window_index:04d}"
            image_paths, frame_groups, debug_paths = _write_detection_request_images(
                helpers=helpers,
                video_path=video_path,
                working_frame_rows_by_index=working_frame_rows_by_index,
                frame_indices=window_frame_indices,
                frame_h=frame_h,
                frame_w=frame_w,
                output_dir=request_dir,
                max_images_per_request=int(args.missed_creatures_max_images_per_request),
                collage_cols=int(args.collage_cols),
                collage_tile_max_edge=int(args.collage_tile_max_edge),
            )
            if not image_paths:
                continue

            parsed, raw_text = _request_detection_issues(
                helpers=helpers,
                send_generate_request_fn=send_generate_request_fn,
                system_prompt=detection_system_prompt,
                image_paths=image_paths,
                frame_groups=frame_groups,
                window_frame_indices=window_frame_indices,
                max_issues=int(args.max_missed_creature_issues_per_window),
                max_json_retries=max_json_retries,
            )
            issues = _sanitize_detection_response(
                helpers=helpers,
                parsed=parsed,
                allowed_frame_indices=set(int(x) for x in window_frame_indices),
                frame_w=frame_w,
                frame_h=frame_h,
                max_issues=int(args.max_missed_creature_issues_per_window),
            )
            report["detection_requests"].append(
                {
                    "round_index": int(round_index),
                    "window_index": int(window_index),
                    "window_frame_indices": [int(x) for x in window_frame_indices],
                    "request_image_paths": image_paths,
                    "request_frame_groups": frame_groups,
                    "debug_image_paths": debug_paths,
                    "raw_response": raw_text,
                    "issue_candidates": issues,
                }
            )

            for issue_index, issue in enumerate(issues):
                target_frame_index = int(issue["target_frame_index"])
                if target_frame_index in invalid_frame_indices:
                    continue

                issue_dir = request_dir / f"issue_{issue_index:04d}_frame_{target_frame_index:06d}"
                issue_dir.mkdir(parents=True, exist_ok=True)
                target_frame_bgr = helpers.read_video_frame(video_path, target_frame_index)
                if target_frame_bgr is None:
                    report["issues"].append(
                        {
                            "round_index": int(round_index),
                            "window_index": int(window_index),
                            "target_frame_index": int(target_frame_index),
                            "description": issue.get("description", ""),
                            "confidence": float(issue.get("confidence", 0.0)),
                            "status": "unresolved",
                            "failure_reason": "target_frame_unreadable",
                            "attempts": [],
                        }
                    )
                    continue

                positive_points = helpers.dedupe_points(
                    [(int(point[0]), int(point[1])) for point in issue.get("click_points", [])],
                    min_distance=4.0,
                )
                issue_report: dict[str, Any] = {
                    "round_index": int(round_index),
                    "window_index": int(window_index),
                    "target_frame_index": int(target_frame_index),
                    "window_frame_indices": [int(x) for x in window_frame_indices],
                    "description": issue.get("description", ""),
                    "confidence": float(issue.get("confidence", 0.0)),
                    "initial_click_points": [[int(x), int(y)] for x, y in positive_points],
                    "status": "unresolved",
                    "attempts": [],
                }

                accepted = False
                for attempt_index in range(1, max(1, int(args.missed_creatures_max_attempts)) + 1):
                    attempt_dir = issue_dir / f"attempt_{attempt_index:02d}"
                    attempt_dir.mkdir(parents=True, exist_ok=True)

                    backend.reset_session(session_id)
                    response = backend.add_point_prompt(
                        session_id=session_id,
                        frame_idx=int(target_frame_index),
                        obj_id=1,
                        points=[(float(x), float(y), 1) for x, y in positive_points],
                        frame_size=(frame_w, frame_h),
                    )
                    attempt_outputs = helpers.unwrap_backend_outputs(response)
                    candidate_item = helpers.select_point_prompt_candidate(
                        frame_index=target_frame_index,
                        outputs=attempt_outputs,
                        requested_obj_id=1,
                        frame_h=frame_h,
                        frame_w=frame_w,
                    )
                    current_row = working_frame_rows_by_index.get(
                        target_frame_index,
                        _empty_frame_row(target_frame_index),
                    )
                    current_items = helpers.decode_frame_row_masks(
                        current_row,
                        frame_h=frame_h,
                        frame_w=frame_w,
                    )
                    max_existing_iou = max(
                        (
                            helpers.binary_mask_iou(candidate_item["mask"], existing_item["mask"])
                            for existing_item in current_items
                        ),
                        default=0.0,
                    ) if candidate_item is not None else 0.0

                    verification_collage_path = str(attempt_dir / "verification_collage.jpg")
                    _build_verification_collage(
                        helpers=helpers,
                        video_path=video_path,
                        target_frame_index=target_frame_index,
                        target_frame_bgr=target_frame_bgr,
                        working_frame_rows_by_index=working_frame_rows_by_index,
                        current_items=current_items,
                        positive_points=positive_points,
                        candidate_item=candidate_item,
                        output_path=verification_collage_path,
                    )
                    attempt_history_text = _build_attempt_history_text(issue_report["attempts"])
                    verdict, raw_verdict = _request_candidate_verdict(
                        helpers=helpers,
                        send_generate_request_fn=send_generate_request_fn,
                        system_prompt=verify_system_prompt,
                        collage_path=verification_collage_path,
                        target_frame_index=target_frame_index,
                        issue_description=str(issue.get("description", "")),
                        positive_points=positive_points,
                        max_existing_iou=max_existing_iou,
                        attempt_history_text=attempt_history_text,
                        candidate_present=(candidate_item is not None),
                        max_json_retries=max_json_retries,
                        frame_w=frame_w,
                        frame_h=frame_h,
                    )

                    attempt_report = {
                        "attempt_index": int(attempt_index),
                        "positive_points": [[int(x), int(y)] for x, y in positive_points],
                        "candidate_present": bool(candidate_item is not None),
                        "candidate_bbox_xyxy": (
                            list(candidate_item["bbox_xyxy"])
                            if candidate_item is not None and candidate_item.get("bbox_xyxy") is not None
                            else None
                        ),
                        "candidate_area": int(candidate_item.get("area") or 0) if candidate_item else 0,
                        "candidate_score": float(candidate_item.get("score") or 0.0) if candidate_item else 0.0,
                        "max_existing_iou": float(max_existing_iou),
                        "verification_collage_path": verification_collage_path,
                        "raw_verdict_response": raw_verdict,
                        **verdict,
                    }
                    issue_report["attempts"].append(attempt_report)

                    if candidate_item is None and verdict["decision"] == "accept":
                        verdict["decision"] = "retry"
                        attempt_report["decision"] = "retry"
                        attempt_report["reason"] = (
                            f"{attempt_report['reason']} " if attempt_report["reason"] else ""
                        ) + "No candidate mask was produced."

                    if (
                        candidate_item is not None
                        and max_existing_iou >= float(args.missed_creatures_duplicate_iou_threshold)
                        and verdict["decision"] == "accept"
                    ):
                        verdict["decision"] = "reject"
                        attempt_report["decision"] = "reject"
                        attempt_report["reason"] = (
                            f"{attempt_report['reason']} " if attempt_report["reason"] else ""
                        ) + "Rejected automatically because the candidate nearly duplicates an existing mask."

                    if verdict["decision"] == "accept" and candidate_item is not None:
                        propagation_result = _append_propagated_masks(
                            helpers=helpers,
                            backend=backend,
                            session_id=session_id,
                            frame_h=frame_h,
                            frame_w=frame_w,
                            working_frame_rows_by_index=working_frame_rows_by_index,
                            target_frame_index=target_frame_index,
                            accepted_mask_item=candidate_item,
                            invalid_frame_indices=invalid_frame_indices,
                            duplicate_iou_threshold=float(args.missed_creatures_duplicate_iou_threshold),
                        )
                        issue_report["status"] = "accepted"
                        issue_report["accepted_attempt_index"] = int(attempt_index)
                        issue_report["accepted_click_points"] = [
                            [int(x), int(y)] for x, y in positive_points
                        ]
                        issue_report["propagation"] = propagation_result
                        accepted_this_round += 1
                        accepted = True
                        break

                    if verdict["decision"] == "retry":
                        new_points = helpers.dedupe_points(
                            positive_points
                            + [
                                (int(point[0]), int(point[1]))
                                for point in verdict.get("additional_points", [])
                            ],
                            min_distance=4.0,
                        )
                        if len(new_points) == len(positive_points):
                            issue_report["failure_reason"] = "retry_without_new_points"
                            break
                        positive_points = new_points
                        continue

                    issue_report["failure_reason"] = (
                        "candidate_rejected"
                        if verdict["decision"] == "reject"
                        else "candidate_not_accepted"
                    )
                    break

                if not accepted and "failure_reason" not in issue_report:
                    issue_report["failure_reason"] = "max_attempts_exhausted"
                report["issues"].append(issue_report)

        if accepted_this_round <= 0:
            break

    report["accepted_issue_count"] = sum(1 for issue in report["issues"] if issue.get("status") == "accepted")
    report["unresolved_issue_count"] = sum(
        1 for issue in report["issues"] if issue.get("status") != "accepted"
    )
    report["failure_reason_counts"] = dict(
        Counter(str(issue.get("failure_reason")) for issue in report["issues"] if issue.get("failure_reason"))
    )
    report["verification_status_counts"] = dict(
        Counter(
            str(attempt.get("decision"))
            for issue in report["issues"]
            for attempt in issue.get("attempts", [])
            if attempt.get("decision")
        )
    )
    helpers.write_json(output_dir / "missed_creatures_report.json", report)
    return report
