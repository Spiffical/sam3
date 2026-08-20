#!/usr/bin/env python3
"""Build strict, frame-grounded SeaTube annotation matching manifests.

This script intentionally makes no MLLM calls.  It extracts the configured
video frame, keeps only annotations whose mapped archive file truly contains
their timestamp, resolves ONC-internal taxonomy IDs, and optionally attaches
numbered SAM3 masks/crops from an existing first-pass or custom-flow run.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


ONC_BASE_URL = "https://data.oceannetworks.ca"
PLACEHOLDER_TAXON_LABELS = {"", "taxon"}
COLORS_BGR = [
    (40, 220, 40),
    (210, 80, 220),
    (40, 190, 240),
    (230, 150, 40),
    (60, 80, 235),
    (220, 220, 50),
    (170, 80, 240),
    (60, 210, 180),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/seatube_exact_frame_samples.json",
        help="Fixed sample specification JSON.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--segmentation-root",
        help=(
            "Optional run directory containing <sample-id>/final_masks_rle.json "
            "or <sample-id>/frame_outputs_rle.json."
        ),
    )
    parser.add_argument(
        "--offline-taxonomy",
        action="store_true",
        help="Do not query ONC taxonomy; use cached/exported labels only.",
    )
    parser.add_argument(
        "--sample-id",
        action="append",
        default=[],
        help="Build only one configured sample; repeat to select several.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def parse_iso_utc(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def to_iso_utc(value: datetime) -> str:
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def git_sha(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def load_annotation_rows(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("annotations"), list):
        rows = payload["annotations"]
    else:
        raise ValueError(f"unsupported annotation export shape: {path}")
    return [row for row in rows if isinstance(row, dict)]


def archive_rows(
    rows: Iterable[dict[str, Any]], archive_filename: str
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if str(row.get("archiveFilename") or "") == archive_filename
    ]


def is_worms_taxon(taxon: dict[str, Any]) -> bool:
    try:
        if int(taxon.get("taxonomyId")) == 1:
            return True
    except (TypeError, ValueError):
        pass
    return str(taxon.get("taxonomyCode") or "").strip().lower() == "worms"


def worms_taxa(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        taxon
        for taxon in (row.get("taxonomy") or row.get("taxons") or [])
        if isinstance(taxon, dict) and is_worms_taxon(taxon)
    ]


def annotation_creator(row: dict[str, Any]) -> dict[str, Any]:
    creator = row.get("createdBy")
    if not isinstance(creator, dict):
        creator = {}
    raw_user_id = creator.get("userId", row.get("creator_user_id"))
    try:
        user_id = int(raw_user_id) if raw_user_id is not None else None
    except (TypeError, ValueError):
        user_id = None
    first_name = str(
        creator.get("firstName") or row.get("creator_first_name") or ""
    ).strip()
    last_name = str(
        creator.get("lastName") or row.get("creator_last_name") or ""
    ).strip()
    return {
        "user_id": user_id,
        "name": " ".join(part for part in (first_name, last_name) if part) or None,
    }


def creator_user_id(row: dict[str, Any]) -> int | None:
    return annotation_creator(row)["user_id"]


def strict_row_relative_seconds(row: dict[str, Any]) -> tuple[float, float]:
    annotation_time = parse_iso_utc(str(row["startDate"]))
    clip_start = parse_iso_utc(str(row["archiveClipStartDate"]))
    duration = float(row["clipDurationSeconds"])
    relative = (annotation_time - clip_start).total_seconds()
    if relative < 0.0 or relative >= duration:
        raise ValueError(
            f"annotation {row.get('annotationId')} lies outside mapped clip: "
            f"relative={relative:.3f}s duration={duration:.3f}s"
        )
    return relative, duration


class TaxonResolver:
    def __init__(self, cache_path: Path, *, offline: bool) -> None:
        self.cache_path = cache_path
        self.offline = offline
        self.cache: dict[str, Any] = {}
        if cache_path.exists():
            payload = read_json(cache_path)
            if isinstance(payload, dict):
                self.cache = payload
        self.changed = False

    @staticmethod
    def key(taxonomy_id: int, taxon_id: int) -> str:
        return f"{taxonomy_id}:{taxon_id}"

    def resolve(self, taxonomy_id: int, taxon_id: int) -> dict[str, Any] | None:
        key = self.key(taxonomy_id, taxon_id)
        if key in self.cache:
            cached = self.cache[key]
            return cached if isinstance(cached, dict) else None
        if self.offline:
            return None

        url = (
            f"{ONC_BASE_URL}/internal/taxonomies/{taxonomy_id}/taxons/{taxon_id}"
        )
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "SAM3-SeaTube-Autolabeling/1.0",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = json.load(response)
            if not isinstance(payload, dict):
                payload = None
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            print(
                f"[WARN] taxonomy {taxonomy_id}:{taxon_id} lookup failed: {exc}",
                flush=True,
            )
            payload = None
        self.cache[key] = payload
        self.changed = True
        return payload

    def save(self) -> None:
        if self.changed:
            write_json(self.cache_path, self.cache)


def clean_export_label(value: Any) -> str | None:
    label = str(value or "").strip()
    return label if label.lower() not in PLACEHOLDER_TAXON_LABELS else None


def resolve_taxon(
    taxon: dict[str, Any], resolver: TaxonResolver
) -> dict[str, Any]:
    taxonomy_id = int(taxon["taxonomyId"]) if taxon.get("taxonomyId") else None
    taxon_id = int(taxon["taxonId"]) if taxon.get("taxonId") else None
    export_label = clean_export_label(taxon.get("displayText"))
    result: dict[str, Any] = {
        "taxonomy_code": taxon.get("taxonomyCode"),
        "taxonomy_id": taxonomy_id,
        "onc_taxon_id": taxon_id,
        "display_name": export_label,
        "resolution_status": "export_label" if export_label else "unresolved",
        "reference_url": taxon.get("taxonUrl"),
        "attributes": taxon.get("attributes") or [],
    }
    if taxonomy_id is None or taxon_id is None:
        return result

    record = resolver.resolve(taxonomy_id, taxon_id)
    if not record:
        return result
    external = record.get("jsonTaxonData") or {}
    result.update(
        {
            "display_name": (
                record.get("commonName")
                or external.get("scientificname")
                or external.get("valid_name")
                or export_label
            ),
            "resolution_status": "onc_taxonomy",
            "reference_id": record.get("referenceId"),
            "reference_url": record.get("referenceUrl") or external.get("url"),
            "scientific_name": external.get("scientificname"),
            "valid_name": external.get("valid_name"),
            "rank": external.get("rank"),
            "status": external.get("status"),
            "aphia_id": external.get("AphiaID"),
            "english_names": record.get("englishNames") or [],
        }
    )
    return result


def annotation_manifest_record(
    row: dict[str, Any],
    *,
    frame_time_utc: datetime,
    relative_seconds: float,
    resolver: TaxonResolver,
) -> dict[str, Any]:
    annotation_time = parse_iso_utc(str(row["startDate"]))
    return {
        "annotation_id": int(row["annotationId"]),
        "annotation_source": row.get("annotationSource"),
        "annotation_start_utc": to_iso_utc(annotation_time),
        "annotation_end_utc": row.get("endDate"),
        "annotation_relative_seconds": round(relative_seconds, 6),
        "delta_from_frame_seconds": round(
            (annotation_time - frame_time_utc).total_seconds(), 6
        ),
        "comment": str(row.get("comment") or "").strip(),
        "creator": annotation_creator(row),
        "taxa": [
            resolve_taxon(taxon, resolver)
            for taxon in worms_taxa(row)
        ],
        "review": {
            "to_be_reviewed": row.get("toBeReviewed"),
            "positive_reviews": row.get("numPositiveReviews"),
            "total_reviews": row.get("numTotalReviews"),
        },
        "contextual_link": row.get("contextualLink"),
    }


def extract_frame(video_path: Path, target_seconds: float) -> tuple[np.ndarray, dict[str, Any]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0.0 or frame_count <= 0:
        capture.release()
        raise RuntimeError(f"invalid video metadata: {video_path}")
    frame_index = int(round(target_seconds * fps))
    frame_index = min(max(0, frame_index), frame_count - 1)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"could not read frame {frame_index}: {video_path}")
    actual_seconds = frame_index / fps
    return frame, {
        "fps": fps,
        "frame_count": frame_count,
        "frame_index": frame_index,
        "requested_time_seconds": target_seconds,
        "decoded_time_seconds": actual_seconds,
        "time_error_seconds": actual_seconds - target_seconds,
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
    }


def write_single_frame_video(path: Path, frame: np.ndarray, fps: float) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (int(frame.shape[1]), int(frame.shape[0])),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not create one-frame video: {path}")
    writer.write(frame)
    writer.release()


def decode_rle(rle: dict[str, Any]) -> np.ndarray:
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise RuntimeError(
            "pycocotools is required when --segmentation-root is used"
        ) from exc
    normalized = dict(rle)
    counts = normalized.get("counts")
    if isinstance(counts, str):
        normalized["counts"] = counts.encode("utf-8")
    return np.asarray(mask_utils.decode(normalized)).astype(bool)


def load_segmentation_masks(
    segmentation_root: Path, sample_id: str
) -> tuple[list[np.ndarray], str, Path] | None:
    sample_dir = segmentation_root / sample_id
    custom_path = sample_dir / "final_masks_rle.json"
    if custom_path.exists():
        payload = read_json(custom_path)
        return (
            [decode_rle(rle) for rle in payload.get("masks", [])],
            "custom_final_masks",
            custom_path,
        )

    firstpass_path = sample_dir / "frame_outputs_rle.json"
    if firstpass_path.exists():
        payload = read_json(firstpass_path)
        frames = payload.get("frames") or []
        if len(frames) != 1:
            raise ValueError(f"expected one frame in {firstpass_path}")
        return (
            [decode_rle(rle) for rle in frames[0].get("out_binary_masks_rle", [])],
            "firstpass_masks",
            firstpass_path,
        )
    return None


def mask_geometry(mask: np.ndarray) -> tuple[list[int], list[float], tuple[int, int]]:
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError("empty segmentation mask")
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    height, width = mask.shape
    normalized = [x1 / width, y1 / height, x2 / width, y2 / height]
    return [x1, y1, x2, y2], normalized, (int(np.median(xs)), int(np.median(ys)))


def attach_objects(
    frame: np.ndarray,
    masks: list[np.ndarray],
    *,
    sample_dir: Path,
    segmentation_source: str,
    segmentation_path: Path,
) -> tuple[list[dict[str, Any]], str]:
    height, width = frame.shape[:2]
    overlay = frame.copy()
    objects_dir = sample_dir / "objects"
    objects_dir.mkdir(parents=True, exist_ok=True)
    objects: list[dict[str, Any]] = []
    for index, raw_mask in enumerate(masks, 1):
        mask = np.asarray(raw_mask).astype(bool)
        if mask.shape != (height, width):
            raise ValueError(
                f"mask {index} shape {mask.shape} does not match frame {(height, width)}"
            )
        if not mask.any():
            continue
        box, box_norm, anchor = mask_geometry(mask)
        color = COLORS_BGR[(index - 1) % len(COLORS_BGR)]
        tinted = overlay.copy()
        tinted[mask] = color
        overlay = cv2.addWeighted(tinted, 0.30, overlay, 0.70, 0)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color, 2, cv2.LINE_AA)
        cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), color, 2)
        label = str(index)
        cv2.putText(
            overlay,
            label,
            (anchor[0] - 6, anchor[1] + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            label,
            (anchor[0] - 6, anchor[1] + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        pad = max(8, int(round(max(box[2] - box[0], box[3] - box[1]) * 0.12)))
        left, top = max(0, box[0] - pad), max(0, box[1] - pad)
        right, bottom = min(width, box[2] + pad), min(height, box[3] + pad)
        crop = frame[top:bottom, left:right].copy()
        crop_mask = mask[top:bottom, left:right]
        crop_tinted = crop.copy()
        crop_tinted[crop_mask] = color
        crop = cv2.addWeighted(crop_tinted, 0.25, crop, 0.75, 0)
        crop_path = objects_dir / f"object_{index:03d}_crop.png"
        mask_path = objects_dir / f"object_{index:03d}_mask.png"
        if not cv2.imwrite(str(crop_path), crop):
            raise RuntimeError(f"could not write {crop_path}")
        if not cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255):
            raise RuntimeError(f"could not write {mask_path}")
        objects.append(
            {
                "object_id": index,
                "box_xyxy_pixels": box,
                "box_xyxy_normalized": [round(value, 8) for value in box_norm],
                "area_pixels": int(mask.sum()),
                "crop_path": str(crop_path),
                "mask_path": str(mask_path),
                "segmentation_source": segmentation_source,
                "segmentation_path": str(segmentation_path),
            }
        )
    overlay_path = sample_dir / "numbered_objects.png"
    if not cv2.imwrite(str(overlay_path), overlay):
        raise RuntimeError(f"could not write {overlay_path}")
    return objects, str(overlay_path)


def build_sample(
    sample: dict[str, Any],
    *,
    output_root: Path,
    resolver: TaxonResolver,
    segmentation_root: Path | None,
) -> dict[str, Any]:
    sample_id = str(sample["id"])
    sample_dir = output_root / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(str(sample["metadata_file"])).expanduser().resolve()
    video_path = Path(str(sample["video"])).expanduser().resolve()
    archive_filename = str(sample["archive_filename"])
    target_seconds = float(sample["target_time_seconds"])
    tolerance = float(sample.get("annotation_tolerance_seconds", 0.05))
    configured_creator_id = int(sample["creator_user_id"])
    expected_candidate_count = sample.get("expected_annotation_candidates")
    if expected_candidate_count is not None:
        expected_candidate_count = int(expected_candidate_count)

    archive_matches = archive_rows(load_annotation_rows(metadata_path), archive_filename)
    worms_rows = [row for row in archive_matches if worms_taxa(row)]
    rows = [
        row
        for row in worms_rows
        if creator_user_id(row) == configured_creator_id
    ]
    if not rows:
        available_creator_ids = sorted(
            {
                user_id
                for row in worms_rows
                if (user_id := creator_user_id(row)) is not None
            }
        )
        raise ValueError(
            f"no WoRMS metadata rows by creator {configured_creator_id} for "
            f"{archive_filename} in {metadata_path}; available creator IDs: "
            f"{available_creator_ids}"
        )
    frame, video_meta = extract_frame(video_path, target_seconds)
    decoded_seconds = float(video_meta["decoded_time_seconds"])
    clip_starts = {str(row.get("archiveClipStartDate")) for row in rows}
    if len(clip_starts) != 1:
        raise ValueError(f"inconsistent archive clip starts for {archive_filename}")
    clip_start_utc = parse_iso_utc(next(iter(clip_starts)))
    frame_time_utc = clip_start_utc + timedelta(seconds=decoded_seconds)

    candidates: list[dict[str, Any]] = []
    outside_mapping_ids: list[int] = []
    for row in rows:
        try:
            relative, _duration = strict_row_relative_seconds(row)
        except ValueError:
            outside_mapping_ids.append(int(row["annotationId"]))
            continue
        if abs(relative - decoded_seconds) <= tolerance:
            candidates.append(
                annotation_manifest_record(
                    row,
                    frame_time_utc=frame_time_utc,
                    relative_seconds=relative,
                    resolver=resolver,
                )
            )
    candidates.sort(
        key=lambda row: (row["annotation_start_utc"], row["annotation_id"])
    )
    if not candidates:
        raise ValueError(
            f"no WoRMS candidates by creator {configured_creator_id} within "
            f"{tolerance:.3f}s of frame {decoded_seconds:.3f}s"
        )
    if (
        expected_candidate_count is not None
        and len(candidates) != expected_candidate_count
    ):
        raise ValueError(
            f"expected {expected_candidate_count} annotation candidates for "
            f"{sample_id}, found {len(candidates)}"
        )

    frame_path = sample_dir / "frame.png"
    one_frame_video_path = sample_dir / "frame.mp4"
    if not cv2.imwrite(str(frame_path), frame):
        raise RuntimeError(f"could not write {frame_path}")
    write_single_frame_video(one_frame_video_path, frame, float(video_meta["fps"]))

    objects: list[dict[str, Any]] = []
    numbered_overlay: str | None = None
    if segmentation_root is not None:
        loaded = load_segmentation_masks(segmentation_root, sample_id)
        if loaded is not None:
            masks, source, source_path = loaded
            objects, numbered_overlay = attach_objects(
                frame,
                masks,
                sample_dir=sample_dir,
                segmentation_source=source,
                segmentation_path=source_path,
            )

    annotation_ids = [row["annotation_id"] for row in candidates]
    object_ids = [row["object_id"] for row in objects]
    timestamp_groups: dict[str, int] = {}
    for row in candidates:
        timestamp = str(row["annotation_start_utc"])
        timestamp_groups[timestamp] = timestamp_groups.get(timestamp, 0) + 1
    for row in candidates:
        row["same_timestamp_annotation_count"] = timestamp_groups[
            str(row["annotation_start_utc"])
        ]

    manifest = {
        "schema_version": 1,
        "sample_id": sample_id,
        "difficulty": sample.get("difficulty"),
        "visual_note": sample.get("visual_note"),
        "source": {
            "metadata_file": str(metadata_path),
            "video": str(video_path),
            "archive_filename": archive_filename,
            "archive_clip_start_utc": to_iso_utc(clip_start_utc),
            "strict_timestamp_containment": True,
            "creator_user_id": configured_creator_id,
            "outside_mapping_annotation_ids": sorted(outside_mapping_ids),
        },
        "frame": {
            **video_meta,
            "frame_time_utc": to_iso_utc(frame_time_utc),
            "frame_path": str(frame_path),
            "one_frame_video_path": str(one_frame_video_path),
        },
        "annotation_candidate_rule": {
            "type": "absolute_timestamp_delta",
            "tolerance_seconds": tolerance,
            "taxonomy_filter": {"taxonomy_code": "WoRMS", "taxonomy_id": 1},
            "creator_filter": {"user_id": configured_creator_id},
            "expected_candidate_count": expected_candidate_count,
            "candidate_count": len(candidates),
        },
        "annotation_candidates": candidates,
        "objects": objects,
        "numbered_objects_path": numbered_overlay,
        "matching": {
            "status": "not_run",
            "allow_one_annotation_to_many_objects": True,
            "allow_one_object_to_many_annotations": False,
            "matches": [],
            "unmatched_object_ids": object_ids,
            "unmatched_annotation_ids": annotation_ids,
            "instructions": (
                "Match only visibly corresponding segmented creatures to SeaTube "
                "annotation candidates. Preserve explicit unmatched objects and "
                "unmatched annotations; never force a complete assignment."
            ),
        },
    }
    write_json(sample_dir / "match_manifest.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    config_path = Path(args.config).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config = read_json(config_path)
    requested = set(args.sample_id)
    samples = [
        sample
        for sample in config.get("samples", [])
        if not requested or str(sample.get("id")) in requested
    ]
    found = {str(sample.get("id")) for sample in samples}
    if requested - found:
        raise SystemExit(f"unknown sample ids: {sorted(requested - found)}")
    if not samples:
        raise SystemExit("no samples selected")

    resolver = TaxonResolver(
        output_root / "onc_taxonomy_cache.json", offline=args.offline_taxonomy
    )
    segmentation_root = (
        Path(args.segmentation_root).expanduser().resolve()
        if args.segmentation_root
        else None
    )
    manifests = []
    for sample in samples:
        print(f"[{sample['id']}] building strict frame manifest", flush=True)
        manifest = build_sample(
            sample,
            output_root=output_root,
            resolver=resolver,
            segmentation_root=segmentation_root,
        )
        manifests.append(manifest)
        print(
            f"[{sample['id']}] annotations={len(manifest['annotation_candidates'])} "
            f"objects={len(manifest['objects'])}",
            flush=True,
        )
    resolver.save()
    aggregate = {
        "schema_version": 1,
        "benchmark_id": config.get("benchmark_id"),
        "selection_method": config.get("selection_method"),
        "taxonomy_filter": {"taxonomy_code": "WoRMS", "taxonomy_id": 1},
        "generator": str(Path(__file__).resolve()),
        "git_sha": git_sha(repo_root),
        "claude_api_calls": False,
        "samples": manifests,
    }
    write_json(output_root / "manifest.json", aggregate)
    print(f"Wrote {output_root / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
