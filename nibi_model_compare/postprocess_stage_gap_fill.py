from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
from typing import Any


def repair_missing_masks_in_window_stage(
    *,
    args: Any,
    send_generate_request_fn: Any,
    missing_system_prompt: str,
    verify_system_prompt: str,
    backend: Any,
    session_id: str,
    video_path: str,
    frame_h: int,
    frame_w: int,
    window_dir: Path,
    window_frame_indices: list[int],
    working_frame_rows_by_index: dict[int, dict[str, Any]],
    local_ids_by_frame: dict[int, list[int]],
    mask_items_by_frame: dict[int, list[dict[str, Any]]],
    raw_collage_path: str,
    overlay_before_gap_fill_path: str,
    max_json_retries: int,
    helpers: Any,
) -> tuple[dict[str, Any], dict[int, list[int]], dict[int, list[dict[str, Any]]]]:
    report: dict[str, Any] = {
        "enabled": True,
        "raw_detection_response": None,
        "issues": [],
        "accepted_issue_count": 0,
        "already_present_issue_count": 0,
        "unresolved_issue_count": 0,
    }
    gap_dir = window_dir / "gap_fill"
    gap_dir.mkdir(parents=True, exist_ok=True)

    detection_parsed, detection_raw_text = helpers.request_missing_mask_issues(
        send_generate_request_fn=send_generate_request_fn,
        system_prompt=missing_system_prompt,
        raw_collage_path=raw_collage_path,
        overlay_collage_path=overlay_before_gap_fill_path,
        window_frame_indices=window_frame_indices,
        inventory_text=helpers.build_inventory_text(
            window_frame_indices=window_frame_indices,
            local_ids_by_frame=local_ids_by_frame,
        ),
        max_issues=int(args.max_gap_issues_per_window),
        max_json_retries=max_json_retries,
    )
    report["raw_detection_response"] = detection_raw_text
    issues = helpers.sanitize_missing_issue_response(
        detection_parsed,
        window_frame_indices=window_frame_indices,
        local_ids_by_frame=local_ids_by_frame,
        max_issues=int(args.max_gap_issues_per_window),
    )
    report["issue_candidates"] = issues

    for issue_index, issue in enumerate(issues):
        issue_dir = gap_dir / f"issue_{issue_index:04d}"
        issue_dir.mkdir(parents=True, exist_ok=True)
        target_frame_index = int(issue["target_frame_index"])
        target_frame_path = issue_dir / f"target_frame_{target_frame_index:06d}.jpg"
        reference_collage_path = issue_dir / "reference_collage.jpg"
        issue_report: dict[str, Any] = {
            "issue_index": int(issue_index),
            "target_frame_index": int(target_frame_index),
            "description": issue.get("description", ""),
            "confidence": float(issue.get("confidence", 0.0)),
            "reference_masks": issue.get("reference_masks", []),
            "source_reference_count": int(issue.get("source_reference_count", 1)),
            "source_reference_split_index": int(issue.get("source_reference_split_index", 0)),
            "status": "unresolved",
            "attempts": [],
        }

        target_frame_bgr = helpers.read_video_frame(video_path, target_frame_index)
        if target_frame_bgr is None:
            issue_report["failure_reason"] = "target_frame_unreadable"
            report["issues"].append(issue_report)
            continue
        helpers.cv2.imwrite(str(target_frame_path), target_frame_bgr)

        reference_tiles: list[tuple[int, Any]] = []
        reference_summaries: list[dict[str, Any]] = []
        for ref in issue["reference_masks"]:
            ref_frame_index = int(ref["frame_index"])
            ref_local_id = int(ref["local_id"])
            ref_frame_bgr = helpers.read_video_frame(video_path, ref_frame_index)
            ref_item = helpers.find_mask_item(
                mask_items_by_frame,
                frame_index=ref_frame_index,
                local_id=ref_local_id,
            )
            if ref_frame_bgr is None or ref_item is None:
                continue
            reference_summaries.append(
                {
                    "frame_index": int(ref_frame_index),
                    "local_id": int(ref_local_id),
                    "centroid": list(ref_item["centroid"]) if ref_item.get("centroid") is not None else None,
                    "bbox_xyxy": list(ref_item["bbox_xyxy"]) if ref_item.get("bbox_xyxy") is not None else None,
                    "score": ref_item.get("score"),
                    "area": int(ref_item.get("area") or 0),
                }
            )
            rendered = helpers.draw_mask_focus(
                ref_frame_bgr,
                focus_items=[ref_item],
                existing_items=mask_items_by_frame.get(ref_frame_index, []),
                focus_label_prefix="ref ",
            )
            reference_tiles.append((ref_frame_index, rendered))
        if not reference_tiles:
            issue_report["failure_reason"] = "reference_frames_unreadable"
            report["issues"].append(issue_report)
            continue
        helpers.build_collage(
            reference_tiles,
            str(reference_collage_path),
            cols=min(int(args.collage_cols), max(1, len(reference_tiles))),
            tile_max_edge=int(args.collage_tile_max_edge),
        )
        issue_report["reference_summaries"] = reference_summaries
        issue_report["debug_paths"] = {
            "target_frame_path": str(target_frame_path),
            "reference_collage_path": str(reference_collage_path),
        }

        target_hint_bbox_xyxy = helpers.estimate_target_hint_bbox_xyxy(
            target_frame_index=target_frame_index,
            reference_masks=list(issue["reference_masks"]),
            mask_items_by_frame=mask_items_by_frame,
            frame_w=frame_w,
            frame_h=frame_h,
        )
        issue_report["target_hint_bbox_xyxy"] = (
            list(target_hint_bbox_xyxy) if target_hint_bbox_xyxy is not None else None
        )

        preexisting_match = helpers.find_existing_issue_match(
            target_frame_index=target_frame_index,
            reference_masks=list(issue["reference_masks"]),
            mask_items_by_frame=mask_items_by_frame,
            hint_bbox_xyxy=target_hint_bbox_xyxy,
            min_support_score=float(args.assignment_heuristic_min_score),
        )
        if preexisting_match is not None:
            issue_report["status"] = "already_present"
            issue_report["resolved_local_id"] = int(preexisting_match["local_id"])
            issue_report["resolution_reason"] = (
                "Skipped gap fill because the current working state already contains "
                "a strong matching mask for this creature on the target frame."
            )
            issue_report["existing_match"] = preexisting_match
            report["issues"].append(issue_report)
            continue

        initial_candidates = helpers.build_point_candidates(
            target_frame_index=target_frame_index,
            reference_masks=list(issue["reference_masks"]),
            mask_items_by_frame=mask_items_by_frame,
            frame_w=frame_w,
            frame_h=frame_h,
            max_candidates=int(args.gap_fill_point_candidates),
        )
        issue_report["initial_point_candidates"] = [
            [int(point[0]), int(point[1])] for point in initial_candidates
        ]
        if not initial_candidates:
            issue_report["failure_reason"] = "no_point_candidates"
            report["issues"].append(issue_report)
            continue

        reference_item = None
        for ref in issue["reference_masks"]:
            try:
                ref_frame_index = int(ref["frame_index"])
                ref_local_id = int(ref["local_id"])
            except Exception:
                continue
            reference_item = helpers.find_mask_item(
                mask_items_by_frame,
                frame_index=ref_frame_index,
                local_id=ref_local_id,
            )
            if reference_item is not None:
                break

        pending_points: deque[tuple[int, int]] = deque(initial_candidates)
        seen_points = set(initial_candidates)
        active_positive_points: list[tuple[int, int]] = []
        accepted = False
        for attempt_index in range(1, max(1, int(args.gap_fill_max_attempts)) + 1):
            if not active_positive_points:
                if not pending_points:
                    break
                active_positive_points = [pending_points.popleft()]

            attempt_dir = issue_dir / f"attempt_{attempt_index:02d}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            candidate_overlay_path = attempt_dir / "candidate_overlay.jpg"
            prompt_overlay_path = attempt_dir / "point_prompt_overlay.jpg"
            prompt_points = helpers.build_point_prompt_points(
                positive_points=active_positive_points,
                hint_bbox_xyxy=target_hint_bbox_xyxy,
                frame_w=frame_w,
                frame_h=frame_h,
            )
            attempt_report: dict[str, Any] = {
                "attempt_index": int(attempt_index),
                "point_xy": [
                    int(active_positive_points[-1][0]),
                    int(active_positive_points[-1][1]),
                ],
                "positive_points": [
                    [int(point[0]), int(point[1])] for point in active_positive_points
                ],
                "prompt_points": [
                    {"x": int(x), "y": int(y), "label": int(label)}
                    for x, y, label in prompt_points
                ],
                "candidate_overlay_path": str(candidate_overlay_path),
                "point_prompt_overlay_path": str(prompt_overlay_path),
            }

            target_existing_items = helpers.decode_frame_row_masks(
                working_frame_rows_by_index[target_frame_index],
                frame_h=frame_h,
                frame_w=frame_w,
            )
            rendered_prompt = helpers.render_point_prompt_debug(
                target_frame_bgr,
                prompt_points=prompt_points,
                hint_bbox_xyxy=target_hint_bbox_xyxy,
                existing_items=target_existing_items,
            )
            helpers.cv2.imwrite(str(prompt_overlay_path), rendered_prompt)

            backend.reset_session(session_id)
            response = backend.add_point_prompt(
                session_id=session_id,
                frame_idx=target_frame_index,
                obj_id=1,
                points=[
                    (float(x), float(y), int(label))
                    for x, y, label in prompt_points
                ],
                frame_size=(frame_w, frame_h),
            )
            backend_outputs = helpers.unwrap_backend_outputs(response)
            attempt_report["backend_output_summary"] = helpers.summarize_backend_outputs(
                frame_index=target_frame_index,
                outputs=backend_outputs,
                frame_h=frame_h,
                frame_w=frame_w,
            )
            candidate_item = helpers.select_point_prompt_candidate(
                frame_index=target_frame_index,
                outputs=backend_outputs,
                requested_obj_id=1,
                frame_h=frame_h,
                frame_w=frame_w,
            )
            if candidate_item is None:
                attempt_report["decision"] = "retry"
                attempt_report["verification_status"] = "not_run"
                attempt_report["failure_reason"] = "no_mask_from_point_prompt"
                issue_report["attempts"].append(attempt_report)
                continue

            max_existing_iou = 0.0
            for existing_item in target_existing_items:
                max_existing_iou = max(
                    max_existing_iou,
                    helpers.binary_mask_iou(candidate_item["mask"], existing_item["mask"]),
                )
            attempt_report["max_existing_iou"] = float(max_existing_iou)
            attempt_report["candidate_bbox_xyxy"] = (
                list(candidate_item["bbox_xyxy"]) if candidate_item.get("bbox_xyxy") is not None else None
            )
            attempt_report["candidate_area"] = int(candidate_item.get("area") or 0)
            attempt_report["candidate_score"] = candidate_item.get("score")

            rendered_candidate = helpers.draw_mask_focus(
                target_frame_bgr,
                focus_items=[candidate_item],
                existing_items=target_existing_items,
                focus_label_prefix="candidate ",
            )
            helpers.cv2.putText(
                rendered_candidate,
                f"clicks={[list(point) for point in active_positive_points]}",
                (12, max(32, rendered_candidate.shape[0] - 18)),
                helpers.cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                helpers.cv2.LINE_AA,
            )
            helpers.cv2.putText(
                rendered_candidate,
                f"clicks={[list(point) for point in active_positive_points]}",
                (12, max(32, rendered_candidate.shape[0] - 18)),
                helpers.cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                1,
                helpers.cv2.LINE_AA,
            )
            helpers.cv2.imwrite(str(candidate_overlay_path), rendered_candidate)

            geometry_assessment = helpers.assess_gap_fill_candidate_geometry(
                candidate_item=candidate_item,
                reference_item=reference_item,
                hint_bbox_xyxy=target_hint_bbox_xyxy,
            )
            attempt_report["geometry_assessment"] = geometry_assessment

            if geometry_assessment["status"] == "reject":
                attempt_report["decision"] = "retry"
                attempt_report["verification_status"] = "geometry_reject"
                attempt_report["failure_reason"] = str(
                    geometry_assessment["reason"] or "geometry_reject"
                )
                attempt_report["reason"] = (
                    "Automatically rejected before MLLM verification because the "
                    "candidate geometry is wildly inconsistent with the reference."
                )
                issue_report["attempts"].append(attempt_report)
                active_positive_points = []
                continue

            if geometry_assessment["status"] == "partial":
                refinement_point = helpers.propose_refinement_point(
                    candidate_item=candidate_item,
                    hint_bbox_xyxy=target_hint_bbox_xyxy,
                    reference_item=reference_item,
                    positive_points=active_positive_points,
                    frame_w=frame_w,
                    frame_h=frame_h,
                )
                attempt_report["auto_refinement_point"] = (
                    list(refinement_point) if refinement_point is not None else None
                )
                if refinement_point is not None:
                    active_positive_points = helpers.dedupe_points(
                        active_positive_points + [refinement_point],
                        min_distance=4.0,
                    )
                    seen_points.add(refinement_point)
                    attempt_report["decision"] = "retry"
                    attempt_report["verification_status"] = "geometry_partial"
                    attempt_report["failure_reason"] = str(
                        geometry_assessment["reason"] or "geometry_partial"
                    )
                    attempt_report["reason"] = (
                        "Automatically retrying with an extra positive click because the "
                        "candidate appears to cover only part of the creature."
                    )
                    issue_report["attempts"].append(attempt_report)
                    continue

            verdict_parsed, verdict_raw_text = helpers.request_gap_fill_verdict(
                send_generate_request_fn=send_generate_request_fn,
                system_prompt=verify_system_prompt,
                raw_target_frame_path=str(target_frame_path),
                candidate_overlay_path=str(candidate_overlay_path),
                reference_collage_path=str(reference_collage_path),
                target_frame_index=target_frame_index,
                issue_description=str(issue.get("description", "")),
                attempt_index=attempt_index,
                positive_points=active_positive_points,
                max_json_retries=max_json_retries,
            )
            attempt_report["raw_verdict_response"] = verdict_raw_text
            decision = str((verdict_parsed or {}).get("decision", "retry")).strip().lower()
            suggested_point = helpers.normalize_point_xy(
                (verdict_parsed or {}).get("suggested_point"),
                frame_w=frame_w,
                frame_h=frame_h,
            )
            attempt_report["decision"] = decision
            attempt_report["verification_status"] = decision
            attempt_report["reason"] = str((verdict_parsed or {}).get("reason", "")).strip()
            attempt_report["suggested_point"] = list(suggested_point) if suggested_point else None

            if max_existing_iou >= 0.80 and decision == "accept":
                duplicate_match = helpers.find_existing_issue_match(
                    target_frame_index=target_frame_index,
                    reference_masks=list(issue["reference_masks"]),
                    mask_items_by_frame=mask_items_by_frame,
                    hint_bbox_xyxy=target_hint_bbox_xyxy,
                    candidate_item=candidate_item,
                    min_support_score=float(args.assignment_heuristic_min_score),
                )
                if duplicate_match is not None:
                    attempt_report["decision"] = "already_present"
                    attempt_report["verification_status"] = "already_present"
                    attempt_report["reason"] = (
                        attempt_report["reason"] + " "
                        if attempt_report["reason"]
                        else ""
                    ) + (
                        "Resolved as already present because the accepted candidate nearly "
                        "duplicates an existing matching mask on the target frame."
                    )
                    attempt_report["resolved_local_id"] = int(duplicate_match["local_id"])
                    attempt_report["existing_match"] = duplicate_match
                    issue_report["status"] = "already_present"
                    issue_report["resolved_local_id"] = int(duplicate_match["local_id"])
                    issue_report["resolution_reason"] = attempt_report["reason"]
                    issue_report["existing_match"] = duplicate_match
                    issue_report["attempts"].append(attempt_report)
                    accepted = True
                    break
                decision = "retry"
                attempt_report["decision"] = "retry"
                attempt_report["verification_status"] = "retry_duplicate"
                attempt_report["reason"] = (
                    attempt_report["reason"] + " "
                    if attempt_report["reason"]
                    else ""
                ) + "Forced retry because candidate nearly duplicates an existing mask."

            if decision == "accept":
                added_local_id = helpers.append_mask_to_frame_row(
                    working_frame_rows_by_index[target_frame_index],
                    frame_h=frame_h,
                    frame_w=frame_w,
                    mask_item=candidate_item,
                )
                updated_items = helpers.decode_frame_row_masks(
                    working_frame_rows_by_index[target_frame_index],
                    frame_h=frame_h,
                    frame_w=frame_w,
                )
                mask_items_by_frame[target_frame_index] = updated_items
                local_ids_by_frame[target_frame_index] = [
                    int(item["local_id"]) for item in updated_items
                ]
                attempt_report["accepted_local_id"] = int(added_local_id)
                issue_report["status"] = "accepted"
                issue_report["accepted_local_id"] = int(added_local_id)
                accepted = True
                issue_report["attempts"].append(attempt_report)
                break

            if decision == "retry" and suggested_point is not None:
                if suggested_point not in seen_points:
                    active_positive_points = helpers.dedupe_points(
                        active_positive_points + [suggested_point],
                        min_distance=4.0,
                    )
                    seen_points.add(suggested_point)
                else:
                    active_positive_points = []
            elif decision == "retry":
                active_positive_points = []
            issue_report["attempts"].append(attempt_report)

            if decision == "reject":
                active_positive_points = []

        if not accepted and "failure_reason" not in issue_report:
            issue_report["failure_reason"] = "max_attempts_exhausted"
        report["issues"].append(issue_report)

    report["accepted_issue_count"] = sum(1 for issue in report["issues"] if issue.get("status") == "accepted")
    report["already_present_issue_count"] = sum(
        1 for issue in report["issues"] if issue.get("status") == "already_present"
    )
    report["unresolved_issue_count"] = sum(
        1 for issue in report["issues"] if issue.get("status") not in {"accepted", "already_present"}
    )
    report["failure_reason_counts"] = dict(
        Counter(str(issue.get("failure_reason")) for issue in report["issues"] if issue.get("failure_reason"))
    )
    report["attempt_failure_reason_counts"] = dict(
        Counter(
            str(attempt.get("failure_reason"))
            for issue in report["issues"]
            for attempt in issue.get("attempts", [])
            if attempt.get("failure_reason")
        )
    )
    report["verification_status_counts"] = dict(
        Counter(
            str(attempt.get("verification_status"))
            for issue in report["issues"]
            for attempt in issue.get("attempts", [])
            if attempt.get("verification_status") is not None
        )
    )
    if args.debug:
        helpers.write_json(gap_dir / "gap_fill_report.json", report)
    return report, local_ids_by_frame, mask_items_by_frame
