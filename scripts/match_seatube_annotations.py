#!/usr/bin/env python3
"""Match whole-frame SeaTube annotations to numbered segmentation masks.

The matcher never invents taxonomy.  Claude receives only the candidate WoRMS
annotations already attached to the selected clip, a numbered mask overlay,
the raw target frame, and short temporal context.  It must return explicit
matches plus unmatched objects and annotations.  Multiple independent repeats
are reduced by majority consensus and rendered into presentation-ready images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from nibi_model_compare.frame_output_utils import decode_rle_to_mask
from sam3.agent.client_claude import send_claude_request


COLORS_BGR = [
    (48, 206, 252),
    (100, 220, 80),
    (230, 120, 210),
    (245, 180, 65),
    (80, 160, 245),
    (215, 220, 75),
    (180, 100, 245),
    (70, 220, 190),
    (245, 110, 95),
    (150, 225, 120),
]


def effective_max_tokens(model: str, requested: int) -> int:
    """Leave enough completion room for Sonnet 5's visual reasoning and JSON."""
    if "sonnet-5" in str(model).lower():
        return max(int(requested), 8192)
    return int(requested)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-manifest",
        default="configs/seatube_meagan_five_frames_v1.json",
    )
    parser.add_argument(
        "--selection-manifest",
        default=(
            "/home/sbialek/ONC/seatube-downloader/downloads/"
            "sam3_seatube_matching_v1/meagan_five_clips_v1/selection_manifest.json"
        ),
    )
    parser.add_argument(
        "--segmentation-root",
        required=True,
        help="Directory containing <frame-id>/final_masks_rle.json and target.png.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=2500)
    parser.add_argument(
        "--min-consensus-confidence",
        type=float,
        default=0.55,
        help=(
            "Minimum mean confidence among the repeat responses supporting a "
            "majority taxon assignment (default: 0.55)."
        ),
    )
    parser.add_argument("--temporal-seconds", type=float, default=0.5)
    parser.add_argument(
        "--frame-id",
        action="append",
        default=[],
        help="Run only this frame id; repeat for multiple ids.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        proc = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip() if proc.returncode == 0 else "unknown"

    status = run("status", "--porcelain")
    return {
        "sha": run("rev-parse", "HEAD"),
        "dirty": bool(status and status != "unknown"),
    }


def extract_answer_json(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    tagged = re.findall(r"<answer>\s*(\{.*?\})\s*</answer>", text, re.DOTALL)
    candidates = list(reversed(tagged))
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def load_masks(path: Path) -> tuple[list[np.ndarray], int, int]:
    payload = read_json(path)
    height, width = [int(value) for value in payload["frame_size_hw"]]
    masks = [
        decode_rle_to_mask(rle, height, width).astype(bool)
        for rle in payload.get("masks", [])
    ]
    return masks, height, width


def mask_anchor(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.where(mask)
    if not len(xs):
        return 20, 20
    return int(np.median(xs)), int(np.median(ys))


def draw_badge(
    image: np.ndarray, position: tuple[int, int], text: str, color: tuple[int, int, int]
) -> None:
    x, y = position
    radius = 17
    x = min(max(radius + 2, x), image.shape[1] - radius - 2)
    y = min(max(radius + 2, y), image.shape[0] - radius - 2)
    cv2.circle(image, (x, y), radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(image, (x, y), radius, color, -1, cv2.LINE_AA)
    scale = 0.60 if len(text) <= 2 else 0.48
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 2)
    cv2.putText(
        image,
        text,
        (x - size[0] // 2, y + size[1] // 2),
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        (15, 15, 15),
        2,
        cv2.LINE_AA,
    )


def render_object_map(frame: np.ndarray, masks: list[np.ndarray], path: Path) -> None:
    # Preserve the raw appearance used for taxonomy. Strong translucent fills
    # can make a distant branching colony look like a solid patch and can also
    # create artificial color differences between otherwise similar objects.
    output = frame.copy()
    for index, mask in enumerate(masks, 1):
        color = COLORS_BGR[(index - 1) % len(COLORS_BGR)]
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(output, contours, -1, color, 3, cv2.LINE_AA)
        draw_badge(output, mask_anchor(mask), str(index), color)
    header = output.copy()
    cv2.rectangle(header, (0, 0), (output.shape[1], 42), (0, 0, 0), -1)
    output = cv2.addWeighted(header, 0.72, output, 0.28, 0)
    cv2.putText(
        output,
        f"NUMBERED SEGMENTATION OBJECTS ({len(masks)})",
        (14, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), output):
        raise RuntimeError(f"could not write {path}")


def render_object_identity_sheet(
    frame: np.ndarray,
    masks: list[np.ndarray],
    path: Path,
) -> None:
    """Render unfilled close-ups so taxonomy is not biased by overlay color."""
    cell_width, cell_height = 400, 300
    cells: list[np.ndarray] = []
    height, width = frame.shape[:2]
    for index, mask in enumerate(masks, 1):
        ys, xs = np.nonzero(mask)
        if len(xs):
            left, right = int(xs.min()), int(xs.max()) + 1
            top, bottom = int(ys.min()), int(ys.max()) + 1
            padding = max(20, round(0.18 * max(right - left, bottom - top)))
            left = max(0, left - padding)
            right = min(width, right + padding)
            top = max(0, top - padding)
            bottom = min(height, bottom + padding)
        else:
            left, top, right, bottom = 0, 0, width, height
        crop = frame[top:bottom, left:right].copy()
        crop_mask = mask[top:bottom, left:right]
        scale = min(
            (cell_width - 12) / max(1, crop.shape[1]),
            (cell_height - 44) / max(1, crop.shape[0]),
        )
        resized_size = (
            max(1, round(crop.shape[1] * scale)),
            max(1, round(crop.shape[0] * scale)),
        )
        zoom = cv2.resize(crop, resized_size, interpolation=cv2.INTER_CUBIC)
        zoom_mask = cv2.resize(
            crop_mask.astype(np.uint8),
            resized_size,
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        color = COLORS_BGR[(index - 1) % len(COLORS_BGR)]
        contours, _ = cv2.findContours(
            zoom_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(zoom, contours, -1, color, 2, cv2.LINE_AA)
        cell = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
        cell[38 : 38 + zoom.shape[0], 6 : 6 + zoom.shape[1]] = zoom
        cv2.putText(
            cell,
            f"OBJECT {index} | raw pixels, contour only",
            (8, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        cells.append(cell)
    columns = min(3, max(1, len(cells)))
    if not cells:
        sheet = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
    else:
        rows = []
        for start in range(0, len(cells), columns):
            row = cells[start : start + columns]
            row.extend(np.zeros_like(cells[0]) for _ in range(columns - len(row)))
            rows.append(np.hstack(row))
        sheet = np.vstack(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), sheet):
        raise RuntimeError(f"could not write {path}")


def read_video_frame(video: Path, frame_index: int) -> tuple[np.ndarray, float, int]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    index = min(max(0, frame_index), max(0, frame_count - 1))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    capture.release()
    if not ok or fps <= 0:
        raise RuntimeError(f"could not decode frame {index} from {video}")
    return frame, fps, index


def render_temporal_triptych(
    video: Path, frame_index: int, temporal_seconds: float, path: Path
) -> None:
    target, fps, _ = read_video_frame(video, frame_index)
    offset = max(1, round(fps * temporal_seconds))
    before, _, before_index = read_video_frame(video, frame_index - offset)
    after, _, after_index = read_video_frame(video, frame_index + offset)
    height, width = target.shape[:2]
    panels = []
    for image, label in (
        (before, f"-{temporal_seconds:.1f}s  f{before_index}"),
        (target, f"TARGET  f{frame_index}"),
        (after, f"+{temporal_seconds:.1f}s  f{after_index}"),
    ):
        panel = cv2.resize(image, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(
            panel,
            label,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(panel)
    triptych = np.hstack(panels)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), triptych):
        raise RuntimeError(f"could not write {path}")


def annotation_name(annotation: dict[str, Any]) -> str:
    return str(annotation["taxon_display_text"]).split(" | ID:")[0]


def build_prompt(
    clip_id: str,
    annotations: list[dict[str, Any]],
    object_count: int,
) -> str:
    lines = []
    for index, annotation in enumerate(annotations, 1):
        count = annotation.get("count")
        count_text = f", recorded count={count}" if count not in (None, "") else ""
        lines.append(
            f"A{index}: annotation_id={annotation['annotation_id']}; "
            f"taxon={annotation_name(annotation)!r}{count_text}; "
            f"reviewed={bool(annotation.get('reviewed'))}"
        )
    candidates = "\n".join(lines) if lines else "(none)"
    return f"""You are matching existing whole-frame SeaTube taxonomy annotations to
numbered SAM3 segmentation objects in an underwater video frame.

FRAME: {clip_id}
NUMBERED OBJECTS: 1 through {object_count}

CANDIDATE ANNOTATIONS (these are the ONLY permitted labels):
{candidates}

Images are supplied in this order:
1. raw target frame;
2. target frame with every segmentation mask outlined and numbered, without
   color fill;
3. raw temporal context before / target / after;
4. one zoomed, contour-only raw-pixel panel for every numbered object.

Rules:
- Match from visual evidence and temporal context. Never invent a taxon.
- Use the raw target and zoomed contour-only panels for morphology; palette
  colors and outlines identify masks but are NOT biological appearance.
- Compare body plan and structural morphology across scale and viewpoint. Depth,
  haze, illumination, focus, and camera white balance can make separate examples
  of one taxon differ substantially in apparent color and contrast. Do not reject
  a match for color or apparent brightness alone. Conversely, proximity, shared
  color, or contact alone is not enough to establish a taxonomic match.
- For branching or colonial life, compare topology, branch thickness and taper,
  surface texture, and repeated growth pattern in the raw close-ups. Separate
  colonies at different depths may share one taxon annotation even though they
  remain separate segmentation objects.
- Inspect the visible morphology of EACH numbered mask independently. A recorded
  count is metadata, not visual evidence and NOT an assignment cap. It may be a
  stale, incomplete, or event-level count. Never label a set merely because its
  size equals the count, and never withhold an otherwise clear visual match only
  because assigning it would exceed the count. Report the disagreement in notes.
- Only group several object IDs under one annotation when every grouped object
  independently shows morphology consistent with that taxon. Otherwise split
  the match or leave the ambiguous objects unmatched.
- A whole-frame annotation can be unmatched if its organism is not segmented or
  not actually visible in this exact frame. A segmentation can be unmatched.
- Never force a complete assignment.
- One annotation may match multiple object IDs when its recorded count or a
  group-level annotation supports that. It may also match multiple independently
  convincing objects when the recorded count is lower or absent—for example,
  three obvious sea stars should all receive the one Asteroidea annotation even
  if its count says 1. Morphologically matching separate coral colonies may share
  one coral annotation. Duplicate annotations may describe the
  same taxon; preserve their IDs instead of guessing which duplicate is unique.
- Do not label rock, sediment, marine snow, vehicle parts, or other non-biological
  material merely because an annotation is available.
- Prefer an explicit unmatched object over a count-based or weak shape guess.
- Use conservative confidence: 0.90+ only for visually clear matches.

Return brief reasoning followed by EXACTLY one JSON object in <answer> tags:
<answer>{{
  "matches": [
    {{
      "object_ids": [1],
      "annotation_ids": [123],
      "confidence": 0.0,
      "reason": "short visual reason"
    }}
  ],
  "unmatched_object_ids": [],
  "unmatched_annotation_ids": [],
  "notes": ["optional uncertainty note"]
}}</answer>
"""


def int_list(value: Any) -> list[int]:
    if isinstance(value, int):
        return [value]
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, int):
            out.append(item)
        elif isinstance(item, str) and item.isdigit():
            out.append(int(item))
    return out


def normalize_response(
    parsed: dict[str, Any] | None,
    *,
    object_count: int,
    annotations: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not isinstance(parsed, dict):
        return None
    allowed_annotations = {int(item["annotation_id"]): item for item in annotations}
    matches = []
    for raw in parsed.get("matches", []):
        if not isinstance(raw, dict):
            continue
        object_ids = int_list(raw.get("object_ids", raw.get("object_id")))
        annotation_ids = int_list(
            raw.get("annotation_ids", raw.get("annotation_id"))
        )
        object_ids = sorted({value for value in object_ids if 1 <= value <= object_count})
        annotation_ids = sorted(
            {value for value in annotation_ids if value in allowed_annotations}
        )
        if not object_ids or not annotation_ids:
            continue
        try:
            confidence = float(raw.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = min(1.0, max(0.0, confidence))
        matches.append(
            {
                "object_ids": object_ids,
                "annotation_ids": annotation_ids,
                "taxa": sorted(
                    {annotation_name(allowed_annotations[value]) for value in annotation_ids}
                ),
                "confidence": confidence,
                "reason": str(raw.get("reason", "")),
            }
        )
    return {
        "matches": matches,
        "unmatched_object_ids": sorted(
            {value for value in int_list(parsed.get("unmatched_object_ids")) if 1 <= value <= object_count}
        ),
        "unmatched_annotation_ids": sorted(
            {
                value
                for value in int_list(parsed.get("unmatched_annotation_ids"))
                if value in allowed_annotations
            }
        ),
        "notes": [str(value) for value in parsed.get("notes", []) if str(value).strip()],
    }


def response_object_choices(response: dict[str, Any]) -> dict[int, dict[str, Any]]:
    choices: dict[int, dict[str, Any]] = {}
    for match in response.get("matches", []):
        for object_id in match["object_ids"]:
            if (
                object_id not in choices
                or match["confidence"] > choices[object_id]["confidence"]
            ):
                choices[object_id] = match
    return choices


def build_consensus(
    responses: list[dict[str, Any]],
    *,
    object_count: int,
    annotations: list[dict[str, Any]],
    configured_repeats: int,
    min_mean_confidence: float = 0.55,
) -> dict[str, Any]:
    min_support = configured_repeats // 2 + 1
    per_object: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for response in responses:
        for object_id, choice in response_object_choices(response).items():
            per_object[object_id].append(choice)

    assignments = []
    used_annotations: set[int] = set()
    for object_id in range(1, object_count + 1):
        rows = per_object.get(object_id, [])
        grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[tuple(row["taxa"])].append(row)
        if not grouped:
            continue
        ranked = sorted(
            grouped.items(),
            key=lambda item: (
                len(item[1]),
                statistics.fmean(row["confidence"] for row in item[1]),
            ),
            reverse=True,
        )
        taxa, supporting = ranked[0]
        support = len(supporting)
        if support < min_support:
            continue
        confidences = [row["confidence"] for row in supporting]
        confidence_mean = statistics.fmean(confidences)
        if confidence_mean < min_mean_confidence:
            continue
        annotation_ids = sorted(
            {
                annotation_id
                for row in supporting
                for annotation_id in row["annotation_ids"]
            }
        )
        used_annotations.update(annotation_ids)
        assignments.append(
            {
                "object_id": object_id,
                "label": " / ".join(taxa),
                "taxa": list(taxa),
                "annotation_ids": annotation_ids,
                "support": support,
                "configured_repeats": configured_repeats,
                "valid_repeats": len(responses),
                "confidence_mean": confidence_mean,
                "confidence_std": (
                    statistics.stdev(confidences) if len(confidences) > 1 else 0.0
                ),
                "reasons": [row["reason"] for row in supporting if row["reason"]],
            }
        )
    assigned_objects = {item["object_id"] for item in assignments}
    all_annotation_ids = {int(item["annotation_id"]) for item in annotations}
    return {
        "consensus_threshold": min_support,
        "min_consensus_confidence": min_mean_confidence,
        "valid_response_count": len(responses),
        "assignments": assignments,
        "unmatched_object_ids": sorted(
            set(range(1, object_count + 1)) - assigned_objects
        ),
        "unmatched_annotation_ids": sorted(all_annotation_ids - used_annotations),
    }


def truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(1, limit - 3)] + "..."


def compress_id_ranges(values: list[int]) -> str:
    ordered = sorted(set(values))
    if not ordered:
        return "none"
    ranges: list[str] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ", ".join(ranges)


def render_labeled_presentation(
    frame: np.ndarray,
    masks: list[np.ndarray],
    consensus: dict[str, Any],
    annotations: list[dict[str, Any]],
    clip_id: str,
    model: str,
    path: Path,
) -> None:
    height, width = frame.shape[:2]
    sidebar_width = 720
    header_height = 52
    assignments = {item["object_id"]: item for item in consensus["assignments"]}

    overlay = frame.copy()
    for index, mask in enumerate(masks, 1):
        color = COLORS_BGR[(index - 1) % len(COLORS_BGR)]
        alpha = 0.48 if index in assignments else 0.20
        overlay[mask] = (
            (1.0 - alpha) * overlay[mask].astype(np.float32)
            + alpha * np.asarray(color)
        ).astype(np.uint8)
    annotated = cv2.addWeighted(overlay, 0.90, frame, 0.10, 0)
    for index, mask in enumerate(masks, 1):
        color = COLORS_BGR[(index - 1) % len(COLORS_BGR)]
        if index not in assignments:
            color = (145, 145, 145)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(annotated, contours, -1, color, 3, cv2.LINE_AA)
        draw_badge(annotated, mask_anchor(mask), str(index), color)

    compact_unmatched = len(masks) > 18
    display_object_ids = (
        sorted(assignments)
        if compact_unmatched
        else list(range(1, len(masks) + 1))
    )
    rows_per_column = max(
        1, math.ceil(max(1, len(display_object_ids)) / 2)
    )
    legend_height = header_height + 54 + rows_per_column * 56 + 125
    canvas_height = max(height + header_height, legend_height)
    canvas = np.full((canvas_height, width + sidebar_width, 3), (24, 27, 31), np.uint8)
    canvas[header_height : header_height + height, :width] = annotated
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], header_height), (8, 10, 13), -1)
    cv2.putText(
        canvas,
        f"Sonnet 5 SeaTube annotation matches | {clip_id}",
        (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    panel_x = width + 18
    cv2.putText(
        canvas,
        "OBJECT  ->  CONSENSUS WoRMS LABEL",
        (panel_x, header_height + 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )
    if compact_unmatched:
        cv2.putText(
            canvas,
            "Matched assignments shown; gray numbered masks are unmatched",
            (panel_x, header_height + 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (175, 185, 195),
            1,
            cv2.LINE_AA,
        )
    column_width = (sidebar_width - 36) // 2
    assignment_start_y = header_height + (94 if compact_unmatched else 72)
    for zero_index, object_id in enumerate(display_object_ids):
        column = zero_index // rows_per_column
        row = zero_index % rows_per_column
        x = panel_x + column * column_width
        y = assignment_start_y + row * 56
        color = COLORS_BGR[(object_id - 1) % len(COLORS_BGR)]
        assignment = assignments.get(object_id)
        if assignment:
            label = truncate(assignment["label"], 38)
            detail = (
                f"{assignment['confidence_mean']:.2f} +/- {assignment['confidence_std']:.2f}"
                f"  ({assignment['support']}/{assignment['configured_repeats']})"
            )
        else:
            color = (145, 145, 145)
            label = "Unmatched / uncertain"
            detail = f"support < {consensus['consensus_threshold']}"
        cv2.rectangle(canvas, (x, y - 22), (x + 34, y + 12), color, -1)
        cv2.putText(
            canvas,
            str(object_id),
            (x + 7, y + 4),
            cv2.FONT_HERSHEY_DUPLEX,
            0.55,
            (15, 15, 15),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (x + 43, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (250, 250, 250),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            detail,
            (x + 43, y + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (175, 185, 195),
            1,
            cv2.LINE_AA,
        )

    annotation_lookup = {int(item["annotation_id"]): item for item in annotations}
    unmatched_names = [
        annotation_name(annotation_lookup[value])
        for value in consensus["unmatched_annotation_ids"]
        if value in annotation_lookup
    ]
    footer_y = assignment_start_y + 16 + rows_per_column * 56
    cv2.putText(
        canvas,
        f"Model: {model} | explicit unmatched objects: {len(consensus['unmatched_object_ids'])}",
        (panel_x, footer_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (175, 185, 195),
        1,
        cv2.LINE_AA,
    )
    if compact_unmatched:
        cv2.putText(
            canvas,
            "Unmatched object IDs: "
            + truncate(
                compress_id_ranges(consensus["unmatched_object_ids"]), 66
            ),
            (panel_x, footer_y + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (175, 185, 195),
            1,
            cv2.LINE_AA,
        )
        annotation_footer_y = footer_y + 54
    else:
        annotation_footer_y = footer_y + 27
    cv2.putText(
        canvas,
        "Unmatched annotations: " + truncate(", ".join(unmatched_names) or "none", 52),
        (panel_x, annotation_footer_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (175, 185, 195),
        1,
        cv2.LINE_AA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"could not write {path}")


def main() -> int:
    args = parse_args()
    if args.repeats < 3:
        raise SystemExit("--repeats must be at least 3 for MLLM matching")
    if not 0.0 <= args.min_consensus_confidence <= 1.0:
        raise SystemExit("--min-consensus-confidence must be between 0 and 1")
    repo_root = Path(__file__).resolve().parents[1]
    # The API key remains in the WSL-only checkout and is never serialized.
    from dotenv import load_dotenv

    load_dotenv(repo_root / ".env")
    benchmark_path = Path(args.benchmark_manifest)
    if not benchmark_path.is_absolute():
        benchmark_path = repo_root / benchmark_path
    benchmark_path = benchmark_path.resolve()
    selection_path = Path(args.selection_manifest).expanduser().resolve()
    segmentation_root = Path(args.segmentation_root).expanduser().resolve()
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    benchmark = read_json(benchmark_path)
    selection = read_json(selection_path)
    selection_by_id = {str(item["clip_id"]): item for item in selection["clips"]}
    requested = set(args.frame_id)
    records = [
        record
        for record in benchmark["frames"]
        if not requested or str(record["id"]) in requested
    ]
    missing = requested - {str(record["id"]) for record in records}
    if missing:
        raise SystemExit(f"frame ids not found: {sorted(missing)}")

    aggregate_rows = []
    max_tokens = effective_max_tokens(args.model, args.max_tokens)
    for record in records:
        clip_id = str(record["id"])
        if clip_id not in selection_by_id:
            raise KeyError(f"selection manifest has no clip {clip_id}")
        annotations = selection_by_id[clip_id]["annotations"]
        segmentation_dir = segmentation_root / clip_id
        masks, height, width = load_masks(segmentation_dir / "final_masks_rle.json")
        target_path = segmentation_dir / "target.png"
        frame = cv2.imread(str(target_path))
        if frame is None or frame.shape[:2] != (height, width):
            raise RuntimeError(f"invalid target image {target_path}")

        frame_output = output_root / clip_id
        object_map_path = frame_output / "object_map.png"
        temporal_path = frame_output / "temporal_context.png"
        identity_sheet_path = frame_output / "object_identity_sheet.png"
        raw_target_path = frame_output / "target.png"
        frame_output.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(raw_target_path), frame):
            raise RuntimeError(f"could not write {raw_target_path}")
        render_object_map(frame, masks, object_map_path)
        render_object_identity_sheet(frame, masks, identity_sheet_path)
        render_temporal_triptych(
            Path(str(record["video"])),
            int(record["frame_index"]),
            args.temporal_seconds,
            temporal_path,
        )
        prompt = build_prompt(clip_id, annotations, len(masks))
        (frame_output / "matching_prompt.txt").write_text(
            prompt, encoding="utf-8"
        )

        normalized_responses = []
        for repeat in range(1, args.repeats + 1):
            repeat_dir = frame_output / f"repeat_{repeat}"
            response_json_path = repeat_dir / "response.json"
            if args.resume and response_json_path.is_file():
                cached = read_json(response_json_path)
                if cached.get("model") == args.model and cached.get("normalized"):
                    print(f"[{clip_id}] repeat {repeat} cached", flush=True)
                    normalized_responses.append(cached["normalized"])
                    continue
            print(f"[{clip_id}] repeat {repeat} matching", flush=True)
            response = send_claude_request(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(raw_target_path)},
                            {"type": "image", "image": str(object_map_path)},
                            {"type": "image", "image": str(temporal_path)},
                            {"type": "image", "image": str(identity_sheet_path)},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                model=args.model,
                max_tokens=max_tokens,
            )
            repeat_dir.mkdir(parents=True, exist_ok=True)
            (repeat_dir / "response.txt").write_text(
                response or "<none>", encoding="utf-8"
            )
            parsed = extract_answer_json(response)
            normalized = normalize_response(
                parsed,
                object_count=len(masks),
                annotations=annotations,
            )
            write_json(
                response_json_path,
                {
                    "model": args.model,
                    "max_tokens": max_tokens,
                    "repeat": repeat,
                    "api_failed": response is None,
                    "parse_failed": normalized is None,
                    "parsed": parsed,
                    "normalized": normalized,
                },
            )
            if normalized is not None:
                normalized_responses.append(normalized)

        consensus = build_consensus(
            normalized_responses,
            object_count=len(masks),
            annotations=annotations,
            configured_repeats=args.repeats,
            min_mean_confidence=args.min_consensus_confidence,
        )
        consensus.update(
            {
                "clip_id": clip_id,
                "model": args.model,
                "segmentation_root": str(segmentation_root),
                "object_count": len(masks),
                "annotation_candidate_count": len(annotations),
                "created_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        write_json(frame_output / "matching_summary.json", consensus)
        render_labeled_presentation(
            frame,
            masks,
            consensus,
            annotations,
            clip_id,
            args.model,
            frame_output / "labeled_presentation.png",
        )
        aggregate_rows.append(consensus)

    aggregate = {
        "schema_version": 1,
        "benchmark_id": benchmark.get("benchmark_id"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "exploratory_not_scored": True,
        "reason_not_scored": "The inherited source checkout is not committed cleanly.",
        "model": args.model,
        "repeats": args.repeats,
        "min_consensus_confidence": args.min_consensus_confidence,
        "whole_frame_review": False,
        "explicit_unmatched_required": True,
        "git": git_state(repo_root),
        "source_hashes": {
            str(Path(__file__).resolve().relative_to(repo_root)): sha256(
                Path(__file__).resolve()
            ),
            str(benchmark_path): sha256(benchmark_path),
            str(selection_path): sha256(selection_path),
        },
        "frames": aggregate_rows,
    }
    write_json(output_root / "summary.json", aggregate)
    print(f"Wrote {output_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
