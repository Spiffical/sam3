#!/usr/bin/env python3
"""Run official FathomNet detectors on a fixed presentation frame set.

This script is deliberately API-free. Model weights are downloaded to the
machine's Hugging Face cache, and all extracted frames and predictions are
written below the requested output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MODELS = {
    "fathomnet_mbari_315k_yolov8": {
        "display_name": "FathomNet MBARI 315k YOLOv8",
        "repo_id": "FathomNet/MBARI-315k-yolov8",
        "filename": "mbari_315k_yolov8.pt",
    },
    "fathomnet_megalodon_2023_yolov8": {
        "display_name": "FathomNet Megalodon 2023 YOLOv8",
        "repo_id": "FathomNet/megalodon-2023-yolov8",
        "filename": "mbari-megalodon-yolov8x.pt",
    },
    "fathomnet_benthic_2025": {
        "display_name": "FathomNet Benthic 2025",
        "repo_id": "FathomNet/2025-MBARI-Benthic-Supercategory-Object-Detector",
        "filename": "best.pt",
    },
    "fathomnet_midwater_2025": {
        "display_name": "FathomNet Midwater 2025",
        "repo_id": "FathomNet/2025-MBARI-Midwater-Supercategory-Object-Detector",
        "filename": "best.pt",
    },
}


@dataclass(frozen=True)
class FrameSpec:
    frame_id: str
    video: Path
    frame_index: int
    time_seconds: float
    visual_note: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/presentation_benchmark_frames.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/presentation_benchmark/fathomnet"),
    )
    parser.add_argument(
        "--run-id",
        help="Output subdirectory name (default: UTC timestamp).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(DEFAULT_MODELS),
        default=sorted(DEFAULT_MODELS),
    )
    parser.add_argument(
        "--imgsz",
        nargs="+",
        type=int,
        default=[640, 1280],
        help="Inference image sizes to compare.",
    )
    parser.add_argument(
        "--render-conf",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.25],
        help="Global confidence thresholds to render from one low-threshold pass.",
    )
    parser.add_argument("--inference-conf", type=float, default=0.01)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--label-mode",
        choices=("classes", "boxes", "none"),
        default="classes",
        help="Per-box label content. Use 'boxes' for taxonomy-neutral slides.",
    )
    return parser.parse_args()


def _load_manifest(path: Path) -> tuple[dict[str, Any], list[FrameSpec]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("frames")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Manifest has no frames: {path}")
    specs: list[FrameSpec] = []
    seen: set[str] = set()
    for row in rows:
        frame_id = str(row["id"])
        if frame_id in seen:
            raise ValueError(f"Duplicate frame id: {frame_id}")
        seen.add(frame_id)
        video = Path(row["video"])
        if not video.is_file():
            raise FileNotFoundError(video)
        specs.append(
            FrameSpec(
                frame_id=frame_id,
                video=video,
                frame_index=int(row["frame_index"]),
                time_seconds=float(row["time_seconds"]),
                visual_note=str(row.get("visual_note", "")),
            )
        )
    return payload, specs


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        proc = subprocess.run(
            ["git", *args], check=False, capture_output=True, text=True
        )
        return proc.stdout.strip() if proc.returncode == 0 else "unknown"

    status = run("status", "--porcelain")
    return {
        "sha": run("rev-parse", "HEAD"),
        "dirty": bool(status and status != "unknown"),
    }


def _extract_frames(specs: list[FrameSpec], frames_dir: Path) -> dict[str, Path]:
    import cv2

    frames_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for spec in specs:
        capture = cv2.VideoCapture(str(spec.video))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        capture.set(cv2.CAP_PROP_POS_FRAMES, spec.frame_index)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError(
                f"Could not read frame {spec.frame_index} from {spec.video}"
            )
        if fps > 0 and abs(spec.frame_index / fps - spec.time_seconds) > 0.05:
            raise ValueError(
                f"Manifest time mismatch for {spec.frame_id}: "
                f"frame/fps={spec.frame_index / fps:.4f}, "
                f"manifest={spec.time_seconds:.4f}"
            )
        output = frames_dir / f"{spec.frame_id}_f{spec.frame_index:03d}.jpg"
        if not cv2.imwrite(str(output), frame, [cv2.IMWRITE_JPEG_QUALITY, 97]):
            raise RuntimeError(f"Could not write {output}")
        paths[spec.frame_id] = output
    return paths


def _result_detections(result: Any) -> list[dict[str, Any]]:
    if result.boxes is None:
        return []
    xyxy = result.boxes.xyxy.detach().cpu().tolist()
    confidence = result.boxes.conf.detach().cpu().tolist()
    classes = result.boxes.cls.detach().cpu().tolist()
    names = result.names
    detections: list[dict[str, Any]] = []
    for box, score, class_index in zip(xyxy, confidence, classes):
        class_id = int(class_index)
        detections.append(
            {
                "xyxy": [round(float(value), 3) for value in box],
                "confidence": round(float(score), 6),
                "class_id": class_id,
                "class_name": str(names[class_id]),
            }
        )
    return detections


def _render_overlay(
    source: Path,
    output: Path,
    detections: list[dict[str, Any]],
    threshold: float,
    title: str,
    label_mode: str,
) -> int:
    import cv2

    frame = cv2.imread(str(source))
    if frame is None:
        raise RuntimeError(f"Could not read {source}")
    kept = [row for row in detections if row["confidence"] >= threshold]
    height, width = frame.shape[:2]
    thickness = max(2, round(min(height, width) / 350))
    font_scale = max(0.45, min(height, width) / 1300)
    color = (40, 225, 90)
    for index, row in enumerate(kept, start=1):
        x1, y1, x2, y2 = (round(value) for value in row["xyxy"])
        x1, x2 = sorted((max(0, x1), min(width - 1, x2)))
        y1, y2 = sorted((max(0, y1), min(height - 1, y2)))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        if label_mode == "classes":
            label = f"{index} {row['class_name']} {row['confidence']:.2f}"
        elif label_mode == "boxes":
            label = str(index)
        else:
            label = ""
        if not label:
            continue
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        label_top = max(0, y1 - text_h - baseline - 5)
        cv2.rectangle(
            frame,
            (x1, label_top),
            (min(width - 1, x1 + text_w + 7), y1),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 3, max(text_h + 1, y1 - baseline - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    header = f"{title} | conf >= {threshold:.2f} | detections: {len(kept)}"
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 38), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    cv2.putText(
        frame,
        header,
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), frame, [cv2.IMWRITE_JPEG_QUALITY, 97]):
        raise RuntimeError(f"Could not write {output}")
    return len(kept)


def _threshold_slug(value: float) -> str:
    return f"conf_{value:.2f}".replace(".", "p")


def _write_gallery(
    run_dir: Path,
    specs: list[FrameSpec],
    gallery_rows: list[dict[str, Any]],
) -> None:
    links: list[str] = []
    for group in gallery_rows:
        name = (
            f"{group['model']}_imgsz{group['imgsz']}_"
            f"{_threshold_slug(group['threshold'])}.html"
        )
        cards = []
        for item in group["items"]:
            relative = item["overlay"].relative_to(run_dir).as_posix()
            note = html.escape(item["visual_note"])
            cards.append(
                f"<figure><img src='../{relative}'>"
                f"<figcaption><b>{html.escape(item['frame_id'])}</b> — "
                f"{item['count']} detections<br>{note}</figcaption></figure>"
            )
        page = f"""<!doctype html><meta charset="utf-8"><title>{name}</title>
<style>body{{font-family:system-ui;background:#111;color:#eee;margin:18px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:18px}}
figure{{margin:0;background:#222;padding:10px}}img{{width:100%;height:auto;display:block}}
figcaption{{padding:8px 2px;line-height:1.4}}a{{color:#8cf}}</style>
<h1>{html.escape(group['model'])} · imgsz {group['imgsz']} · conf {group['threshold']:.2f}</h1>
<div class="grid">{''.join(cards)}</div>\n"""
        gallery_path = run_dir / "galleries" / name
        gallery_path.parent.mkdir(parents=True, exist_ok=True)
        gallery_path.write_text(page, encoding="utf-8")
        links.append(
            f"<li><a href='galleries/{name}'>{html.escape(group['model'])} — "
            f"imgsz {group['imgsz']} — conf {group['threshold']:.2f}</a></li>"
        )

    raw_cards = []
    for spec in specs:
        source = f"frames/{spec.frame_id}_f{spec.frame_index:03d}.jpg"
        raw_cards.append(
            f"<figure><img src='{source}'><figcaption><b>{html.escape(spec.frame_id)}</b> "
            f"frame {spec.frame_index} ({spec.time_seconds:.2f}s)<br>"
            f"{html.escape(spec.visual_note)}</figcaption></figure>"
        )
    index = f"""<!doctype html><meta charset="utf-8"><title>FathomNet benchmark</title>
<style>body{{font-family:system-ui;background:#111;color:#eee;margin:18px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:18px}}
figure{{margin:0;background:#222;padding:10px}}img{{width:100%;height:auto;display:block}}
figcaption{{padding:8px 2px;line-height:1.4}}a{{color:#8cf}}</style>
<h1>Fixed presentation frames</h1><div class="grid">{''.join(raw_cards)}</div>
<h1>FathomNet galleries</h1><ul>{''.join(links)}</ul>\n"""
    (run_dir / "index.html").write_text(index, encoding="utf-8")


def main() -> int:
    args = _parse_args()
    if not 0 < args.inference_conf <= 1:
        raise ValueError("--inference-conf must be in (0, 1]")
    if any(not 0 < value <= 1 for value in args.render_conf):
        raise ValueError("Every --render-conf value must be in (0, 1]")
    if any(value < args.inference_conf for value in args.render_conf):
        raise ValueError("Render thresholds cannot be below --inference-conf")

    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    manifest, specs = _load_manifest(args.manifest)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    frame_paths = _extract_frames(specs, run_dir / "frames")

    metadata: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "exploratory_not_scored": True,
        "manifest": manifest,
        "git": _git_state(),
        "inference": {
            "models": args.models,
            "imgsz": args.imgsz,
            "inference_conf": args.inference_conf,
            "render_conf": args.render_conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "device": args.device,
            "agnostic_nms": True,
            "label_mode": args.label_mode,
        },
        "model_sources": {},
    }
    predictions: list[dict[str, Any]] = []
    gallery_rows: list[dict[str, Any]] = []

    ordered_sources = [str(frame_paths[spec.frame_id]) for spec in specs]
    for model_alias in args.models:
        model_spec = DEFAULT_MODELS[model_alias]
        weights = hf_hub_download(
            repo_id=model_spec["repo_id"], filename=model_spec["filename"]
        )
        metadata["model_sources"][model_alias] = {
            **model_spec,
            "weights_sha256": hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
        }
        model = YOLO(weights)
        for imgsz in args.imgsz:
            results = model.predict(
                source=ordered_sources,
                conf=args.inference_conf,
                iou=args.iou,
                imgsz=imgsz,
                max_det=args.max_det,
                device=args.device,
                agnostic_nms=True,
                verbose=False,
                save=False,
            )
            if len(results) != len(specs):
                raise RuntimeError(
                    f"Expected {len(specs)} results, got {len(results)}"
                )
            parsed: dict[str, list[dict[str, Any]]] = {}
            for spec, result in zip(specs, results):
                rows = _result_detections(result)
                parsed[spec.frame_id] = rows
                predictions.append(
                    {
                        "model": model_alias,
                        "imgsz": imgsz,
                        "frame_id": spec.frame_id,
                        "frame_index": spec.frame_index,
                        "detections": rows,
                    }
                )

            for threshold in sorted(set(args.render_conf)):
                items = []
                for spec in specs:
                    output = (
                        run_dir
                        / "overlays"
                        / model_alias
                        / f"imgsz_{imgsz}"
                        / _threshold_slug(threshold)
                        / f"{spec.frame_id}.jpg"
                    )
                    count = _render_overlay(
                        frame_paths[spec.frame_id],
                        output,
                        parsed[spec.frame_id],
                        threshold,
                        f"{model_spec['display_name']} | {spec.frame_id}",
                        args.label_mode,
                    )
                    items.append(
                        {
                            "frame_id": spec.frame_id,
                            "visual_note": spec.visual_note,
                            "overlay": output,
                            "count": count,
                        }
                    )
                gallery_rows.append(
                    {
                        "model": model_alias,
                        "imgsz": imgsz,
                        "threshold": threshold,
                        "items": items,
                    }
                )

    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "detections.json").write_text(
        json.dumps(predictions, indent=2) + "\n", encoding="utf-8"
    )
    _write_gallery(run_dir, specs, gallery_rows)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
