from __future__ import annotations

from collections import Counter
from typing import Any

import os


def filter_outlier_gap_fill_masks_stage(
    *,
    args: Any,
    send_generate_request_fn: Any,
    outlier_system_prompt: str,
    video_path: str,
    frame_h: int,
    frame_w: int,
    window_dir: Any,
    working_frame_rows_by_index: dict[int, dict[str, Any]],
    local_ids_by_frame: dict[int, list[int]],
    mask_items_by_frame: dict[int, list[dict[str, Any]]],
    gap_fill_report: dict[str, Any] | None,
    max_json_retries: int,
    helpers: Any,
) -> tuple[dict[str, Any], dict[int, list[int]], dict[int, list[dict[str, Any]]]]:
    report: dict[str, Any] = {
        "enabled": True,
        "candidate_issue_count": 0,
        "reviewed_issue_count": 0,
        "kept_mask_count": 0,
        "removed_mask_count": 0,
        "issues": [],
    }
    if not gap_fill_report or not (gap_fill_report.get("issues") or []):
        return report, local_ids_by_frame, mask_items_by_frame

    outlier_dir = window_dir / "outlier_filter"
    outlier_dir.mkdir(parents=True, exist_ok=True)

    accepted_candidates: list[dict[str, Any]] = []
    for issue_index, issue in enumerate(gap_fill_report.get("issues") or []):
        if str(issue.get("status", "")).strip().lower() != "accepted":
            continue
        accepted_local_id = issue.get("accepted_local_id")
        if accepted_local_id is None:
            continue
        try:
            target_frame_index = int(issue.get("target_frame_index"))
            accepted_local_id = int(accepted_local_id)
        except Exception:
            continue
        current_item = helpers.find_mask_item(
            mask_items_by_frame,
            frame_index=target_frame_index,
            local_id=accepted_local_id,
        )
        if current_item is None:
            issue["post_filter_status"] = "skipped_missing_local_id"
            continue
        reference_support_scores: list[float] = []
        for ref in issue.get("reference_masks") or []:
            try:
                ref_frame_index = int(ref["frame_index"])
                ref_local_id = int(ref["local_id"])
            except Exception:
                continue
            ref_item = helpers.find_mask_item(
                mask_items_by_frame,
                frame_index=ref_frame_index,
                local_id=ref_local_id,
            )
            if ref_item is None:
                continue
            reference_support_scores.append(helpers.mask_match_support_score(current_item, ref_item))
        accepted_candidates.append(
            {
                "issue_index": int(issue_index),
                "issue": issue,
                "target_frame_index": int(target_frame_index),
                "accepted_local_id": int(accepted_local_id),
                "current_item": current_item,
                "best_reference_support_score": float(max(reference_support_scores, default=0.0)),
                "reference_support_scores": [float(x) for x in reference_support_scores],
            }
        )

    accepted_candidates.sort(
        key=lambda item: (
            float(item["best_reference_support_score"]),
            float(item["issue"].get("confidence", 0.0)),
            int(item["target_frame_index"]),
            int(item["accepted_local_id"]),
        )
    )
    report["candidate_issue_count"] = len(accepted_candidates)

    max_reviews = max(0, int(args.max_outlier_checks_per_window))
    selected_candidates = accepted_candidates[:max_reviews]
    for skipped in accepted_candidates[max_reviews:]:
        skipped["issue"]["post_filter_status"] = "not_reviewed"
        skipped["issue"]["post_filter_reason"] = "review_budget_exhausted"

    for candidate in selected_candidates:
        issue = candidate["issue"]
        issue_index = int(candidate["issue_index"])
        target_frame_index = int(candidate["target_frame_index"])
        accepted_local_id = int(candidate["accepted_local_id"])
        issue_dir = outlier_dir / f"issue_{issue_index:04d}_frame_{target_frame_index:06d}_l{accepted_local_id:04d}"
        issue_dir.mkdir(parents=True, exist_ok=True)

        target_frame_path = str(issue_dir / f"target_frame_{target_frame_index:06d}.jpg")
        target_frame_bgr = helpers.read_video_frame(video_path, target_frame_index)
        if target_frame_bgr is None:
            issue_report = {
                "issue_index": int(issue_index),
                "target_frame_index": int(target_frame_index),
                "accepted_local_id": int(accepted_local_id),
                "status": "kept",
                "failure_reason": "target_frame_unreadable",
            }
            report["issues"].append(issue_report)
            issue["post_filter_status"] = "kept"
            issue["post_filter_reason"] = "target_frame_unreadable"
            report["kept_mask_count"] += 1
            continue
        helpers.cv2.imwrite(target_frame_path, target_frame_bgr)

        current_item = helpers.find_mask_item(
            mask_items_by_frame,
            frame_index=target_frame_index,
            local_id=accepted_local_id,
        )
        if current_item is None:
            issue_report = {
                "issue_index": int(issue_index),
                "target_frame_index": int(target_frame_index),
                "accepted_local_id": int(accepted_local_id),
                "status": "kept",
                "failure_reason": "accepted_local_id_missing",
            }
            report["issues"].append(issue_report)
            issue["post_filter_status"] = "kept"
            issue["post_filter_reason"] = "accepted_local_id_missing"
            report["kept_mask_count"] += 1
            continue

        reference_collage_path = (
            str((issue.get("debug_paths") or {}).get("reference_collage_path") or "")
        )
        if not reference_collage_path or not os.path.exists(reference_collage_path):
            reference_collage_path = str(issue_dir / "reference_collage.jpg")
            reference_tiles: list[tuple[int, Any]] = []
            for ref in issue.get("reference_masks") or []:
                try:
                    ref_frame_index = int(ref["frame_index"])
                    ref_local_id = int(ref["local_id"])
                except Exception:
                    continue
                ref_frame_bgr = helpers.read_video_frame(video_path, ref_frame_index)
                ref_item = helpers.find_mask_item(
                    mask_items_by_frame,
                    frame_index=ref_frame_index,
                    local_id=ref_local_id,
                )
                if ref_frame_bgr is None or ref_item is None:
                    continue
                reference_tiles.append(
                    (
                        ref_frame_index,
                        helpers.draw_mask_focus(
                            ref_frame_bgr,
                            focus_items=[ref_item],
                            existing_items=mask_items_by_frame.get(ref_frame_index, []),
                            focus_label_prefix="ref ",
                        ),
                    )
                )
            if reference_tiles:
                helpers.build_collage(
                    reference_tiles,
                    reference_collage_path,
                    cols=min(int(args.collage_cols), max(1, len(reference_tiles))),
                    tile_max_edge=int(args.collage_tile_max_edge),
                )
        if not reference_collage_path or not os.path.exists(reference_collage_path):
            issue_report = {
                "issue_index": int(issue_index),
                "target_frame_index": int(target_frame_index),
                "accepted_local_id": int(accepted_local_id),
                "status": "kept",
                "failure_reason": "reference_collage_missing",
            }
            report["issues"].append(issue_report)
            issue["post_filter_status"] = "kept"
            issue["post_filter_reason"] = "reference_collage_missing"
            report["kept_mask_count"] += 1
            continue

        target_existing_items = [
            item
            for item in helpers.decode_frame_row_masks(
                working_frame_rows_by_index[target_frame_index],
                frame_h=frame_h,
                frame_w=frame_w,
            )
            if int(item["local_id"]) != accepted_local_id
        ]
        candidate_overlay_path = str(issue_dir / "candidate_overlay.jpg")
        rendered_candidate = helpers.draw_mask_focus(
            target_frame_bgr,
            focus_items=[current_item],
            existing_items=target_existing_items,
            focus_label_prefix="candidate ",
        )
        helpers.cv2.imwrite(candidate_overlay_path, rendered_candidate)

        verdict_parsed, verdict_raw_text = helpers.request_outlier_mask_verdict(
            send_generate_request_fn=send_generate_request_fn,
            system_prompt=outlier_system_prompt,
            raw_target_frame_path=target_frame_path,
            candidate_overlay_path=candidate_overlay_path,
            reference_collage_path=reference_collage_path,
            target_frame_index=target_frame_index,
            local_id=accepted_local_id,
            issue_description=str(issue.get("description", "")),
            max_json_retries=max_json_retries,
        )
        decision = str((verdict_parsed or {}).get("decision", "keep")).strip().lower()
        if decision not in {"keep", "remove"}:
            decision = "keep"
        reason = str((verdict_parsed or {}).get("reason", "")).strip()

        issue_report = {
            "issue_index": int(issue_index),
            "target_frame_index": int(target_frame_index),
            "accepted_local_id": int(accepted_local_id),
            "status": "kept",
            "reason": reason,
            "raw_verdict_response": verdict_raw_text,
            "candidate_overlay_path": candidate_overlay_path,
            "target_frame_path": target_frame_path,
            "reference_collage_path": reference_collage_path,
            "best_reference_support_score": float(candidate["best_reference_support_score"]),
            "reference_support_scores": list(candidate["reference_support_scores"]),
        }

        if decision == "remove":
            removed = helpers.remove_local_id_from_frame_row(
                working_frame_rows_by_index[target_frame_index],
                local_id=accepted_local_id,
            )
            if removed:
                updated_items = helpers.decode_frame_row_masks(
                    working_frame_rows_by_index[target_frame_index],
                    frame_h=frame_h,
                    frame_w=frame_w,
                )
                mask_items_by_frame[target_frame_index] = updated_items
                local_ids_by_frame[target_frame_index] = [
                    int(item["local_id"]) for item in updated_items
                ]
                issue_report["status"] = "removed"
                report["removed_mask_count"] += 1
                issue["post_filter_status"] = "removed_as_outlier"
                issue["post_filter_reason"] = reason or "Removed by outlier-mask review."
            else:
                issue_report["status"] = "kept"
                issue_report["failure_reason"] = "remove_local_id_failed"
                report["kept_mask_count"] += 1
                issue["post_filter_status"] = "kept"
                issue["post_filter_reason"] = "remove_local_id_failed"
        else:
            report["kept_mask_count"] += 1
            issue["post_filter_status"] = "kept"
            issue["post_filter_reason"] = reason

        report["issues"].append(issue_report)

    report["reviewed_issue_count"] = len(report["issues"])
    report["not_reviewed_issue_count"] = max(0, len(accepted_candidates) - len(report["issues"]))
    report["verdict_counts"] = dict(
        Counter(str(issue.get("status")) for issue in report["issues"] if issue.get("status"))
    )
    report["failure_reason_counts"] = dict(
        Counter(
            str(issue.get("failure_reason"))
            for issue in report["issues"]
            if issue.get("failure_reason")
        )
    )
    if args.debug:
        helpers.write_json(outlier_dir / "outlier_filter_report.json", report)
    return report, local_ids_by_frame, mask_items_by_frame
