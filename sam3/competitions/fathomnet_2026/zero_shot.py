from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from .categories import (
    build_category_name_to_id,
    normalize_category_name,
)
from .dataset import CompetitionDataset, CompetitionImage
from .prompts import (
    DEFAULT_CROP_CLASSIFICATION_PROMPT,
    PRIMARY_DETECTION_PROMPT,
    RESCUE_DETECTION_PROMPTS,
    build_crop_classification_prompt,
)
from . import runtime as runtime_lib
from .submission import DetectionPrediction, write_submission_csv


@dataclass(frozen=True)
class ZeroShotRunConfig:
    server_url: str
    model: str
    api_key: str = ""
    device: str = "cuda"
    checkpoint_path: str = ""
    prompt_profile: str = "fathomnet_2026"
    max_generations: int = 10
    max_completion_tokens: int = 1024
    image_detail: str = "high"
    max_images_per_request: int = 3
    agent_image_max_edge: int = 768
    agent_image_min_edge: int = 384
    confidence_threshold: float = 0.0
    compile_image_model: bool = False
    max_images: int = 0
    proposal_iou_threshold: float = 0.75
    final_iou_threshold: float = 0.55
    crop_context_ratio: float = 0.2
    classification_min_confidence: float = 0.2
    keep_debug_artifacts: bool = False


@dataclass
class DetectionCandidate:
    image_id: int
    image_path: str
    source_prompt: str
    proposal_index: int
    bbox_xywh: tuple[float, float, float, float]
    proposal_score: float
    category_id: Optional[int] = None
    category_name: str = ""
    classification_confidence: float = 0.0
    classification_reason: str = ""
    final_score: float = 0.0
    drop_detection: bool = False
    overlay_path: str = ""
    crop_path: str = ""


def _safe_slug(value: str) -> str:
    safe_chars = []
    for char in str(value):
        if char.isalnum() or char in {"-", "_", "."}:
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    return "".join(safe_chars).strip("._-") or "item"


def _denormalize_bbox_xywh(
    bbox_xywh_norm: Sequence[float],
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    if len(bbox_xywh_norm) != 4:
        raise ValueError(f"Expected normalized bbox with 4 values, got {bbox_xywh_norm}")
    x, y, width, height = [float(value) for value in bbox_xywh_norm]
    return (
        x * float(image_width),
        y * float(image_height),
        width * float(image_width),
        height * float(image_height),
    )


def _bbox_iou(
    bbox_a: Sequence[float],
    bbox_b: Sequence[float],
) -> float:
    ax, ay, aw, ah = [float(value) for value in bbox_a]
    bx, by, bw, bh = [float(value) for value in bbox_b]
    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh
    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    union = max(1e-8, area_a + area_b - inter_area)
    return inter_area / union


def _extract_candidate_json_strings(generated_text: str) -> list[str]:
    candidates: list[str] = []
    decoder = json.JSONDecoder()
    for index, char in enumerate(generated_text):
        if char != "{":
            continue
        try:
            _, end_index = decoder.raw_decode(generated_text[index:])
        except json.JSONDecodeError:
            continue
        payload = generated_text[index : index + end_index].strip()
        if payload:
            candidates.append(payload)
    return list(dict.fromkeys(candidates))


def _parse_classification_response(generated_text: str) -> Optional[dict[str, Any]]:
    if not generated_text:
        return None
    for candidate_str in _extract_candidate_json_strings(generated_text):
        try:
            parsed = json.loads(candidate_str)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        if "drop_detection" not in parsed:
            continue
        return parsed
    return None


def _dedupe_candidates(
    candidates: Iterable[DetectionCandidate],
    *,
    iou_threshold: float,
) -> list[DetectionCandidate]:
    kept: list[DetectionCandidate] = []
    ordered = sorted(
        candidates,
        key=lambda candidate: candidate.proposal_score,
        reverse=True,
    )
    for candidate in ordered:
        duplicate = False
        for existing in kept:
            if candidate.image_id != existing.image_id:
                continue
            if _bbox_iou(candidate.bbox_xywh, existing.bbox_xywh) >= float(iou_threshold):
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def _bbox_to_crop_bounds(
    bbox_xywh: Sequence[float],
    *,
    image_width: int,
    image_height: int,
    context_ratio: float,
) -> tuple[int, int, int, int]:
    x, y, width, height = [float(value) for value in bbox_xywh]
    pad_x = float(context_ratio) * width
    pad_y = float(context_ratio) * height
    x1 = max(0, int(round(x - pad_x)))
    y1 = max(0, int(round(y - pad_y)))
    x2 = min(image_width, int(round(x + width + pad_x)))
    y2 = min(image_height, int(round(y + height + pad_y)))
    if x2 <= x1:
        x2 = min(image_width, x1 + max(1, int(round(width))))
    if y2 <= y1:
        y2 = min(image_height, y1 + max(1, int(round(height))))
    return x1, y1, x2, y2


def _render_overlay_image(
    image_path: str,
    bbox_xywh: Sequence[float],
    output_path: str,
) -> None:
    runtime_lib.ensure_runtime_deps()
    image = runtime_lib.Image.open(image_path).convert("RGB")
    from PIL import ImageDraw

    x, y, width, height = [float(value) for value in bbox_xywh]
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        [x, y, x + width, y + height],
        outline=(255, 128, 0),
        width=4,
    )
    image.save(output_path)


def _save_crop_image(
    image_path: str,
    bbox_xywh: Sequence[float],
    *,
    context_ratio: float,
    output_path: str,
) -> None:
    runtime_lib.ensure_runtime_deps()
    image = runtime_lib.Image.open(image_path).convert("RGB")
    x1, y1, x2, y2 = _bbox_to_crop_bounds(
        bbox_xywh,
        image_width=image.width,
        image_height=image.height,
        context_ratio=context_ratio,
    )
    crop = image.crop((x1, y1, x2, y2))
    crop.save(output_path)


def _classify_candidate(
    *,
    send_request: Any,
    overlay_path: str,
    crop_path: str,
    prompt_text: str,
    category_name_to_id: dict[str, int],
    max_retries: int = 2,
) -> dict[str, Any]:
    last_text = ""
    reminder = ""
    for _ in range(max(1, int(max_retries))):
        messages = [
            {
                "role": "system",
                "content": (
                    prompt_text
                    + "\nReturn JSON only. Do not use markdown or prose outside JSON."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Full image with the detection box for context:",
                    },
                    {"type": "image", "image": overlay_path},
                    {"type": "text", "text": "Detection crop:"},
                    {"type": "image", "image": crop_path},
                    {"type": "text", "text": reminder or "Classify this detection."},
                ],
            },
        ]
        last_text = send_request(messages) or ""
        parsed = _parse_classification_response(last_text)
        if parsed is None:
            reminder = (
                "Your previous response was not valid JSON. "
                "Return exactly one JSON object with keys "
                "category_name, confidence, drop_detection, reason."
            )
            continue

        category_name = normalize_category_name(str(parsed.get("category_name", "")))
        drop_detection = bool(parsed.get("drop_detection", False))
        confidence = float(parsed.get("confidence", 0.0))
        reason = str(parsed.get("reason", "")).strip()

        if drop_detection:
            return {
                "category_id": None,
                "category_name": "",
                "confidence": max(0.0, min(1.0, confidence)),
                "drop_detection": True,
                "reason": reason or "drop_detection=true",
                "raw_text": last_text,
            }

        if category_name not in category_name_to_id:
            reminder = (
                "The category_name must exactly match one allowed label. "
                f"You returned {category_name!r}. Return JSON only."
            )
            continue

        return {
            "category_id": category_name_to_id[category_name],
            "category_name": category_name,
            "confidence": max(0.0, min(1.0, confidence)),
            "drop_detection": False,
            "reason": reason,
            "raw_text": last_text,
        }

    return {
        "category_id": None,
        "category_name": "",
        "confidence": 0.0,
        "drop_detection": True,
        "reason": "classification_parse_failure",
        "raw_text": last_text,
    }


def _final_nms(
    predictions: Iterable[DetectionCandidate],
    *,
    iou_threshold: float,
) -> list[DetectionCandidate]:
    kept: list[DetectionCandidate] = []
    ordered = sorted(
        predictions,
        key=lambda prediction: prediction.final_score,
        reverse=True,
    )
    for prediction in ordered:
        duplicate = False
        for existing in kept:
            if prediction.image_id != existing.image_id:
                continue
            if prediction.category_id != existing.category_id:
                continue
            if _bbox_iou(prediction.bbox_xywh, existing.bbox_xywh) >= float(iou_threshold):
                duplicate = True
                break
        if not duplicate:
            kept.append(prediction)
    return kept


def _prediction_to_json(candidate: DetectionCandidate) -> dict[str, Any]:
    payload = asdict(candidate)
    payload["bbox_xywh"] = [round(value, 5) for value in candidate.bbox_xywh]
    payload["proposal_score"] = round(float(candidate.proposal_score), 8)
    payload["classification_confidence"] = round(
        float(candidate.classification_confidence), 8
    )
    payload["final_score"] = round(float(candidate.final_score), 8)
    return payload


def run_zero_shot_submission(
    *,
    dataset: CompetitionDataset,
    images_dir: str | Path,
    output_dir: str | Path,
    config: ZeroShotRunConfig,
) -> dict[str, Any]:
    runtime_lib.ensure_runtime_deps()

    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    image_runs_root = output_root / "image_runs"
    image_runs_root.mkdir(parents=True, exist_ok=True)

    runtime_lib.configure_agent_environment(
        prompt_profile=config.prompt_profile,
        image_detail=config.image_detail,
        max_images_per_request=config.max_images_per_request,
        agent_image_max_edge=config.agent_image_max_edge,
        agent_image_min_edge=config.agent_image_min_edge,
    )

    send_request = runtime_lib.build_send_request(
        server_url=config.server_url,
        model=config.model,
        api_key=runtime_lib.default_api_key(config.api_key),
        max_completion_tokens=config.max_completion_tokens,
    )

    bpe_path = runtime_lib.find_bpe_path()
    image_model = runtime_lib.build_sam3_image_model(
        bpe_path=bpe_path,
        device=str(config.device),
        checkpoint_path=(config.checkpoint_path or None),
        compile=bool(config.compile_image_model),
    )
    image_processor = runtime_lib.Sam3Processor(
        image_model,
        confidence_threshold=float(config.confidence_threshold),
    )
    local_service = runtime_lib.LocalSam3Service(image_processor)

    images = list(dataset.images)
    if int(config.max_images) > 0:
        images = images[: int(config.max_images)]

    category_name_to_id = build_category_name_to_id(dataset.categories)
    classification_prompt = build_crop_classification_prompt(
        [category.name for category in dataset.categories]
    ) or DEFAULT_CROP_CLASSIFICATION_PROMPT

    raw_candidates_path = output_root / "raw_candidates.jsonl"
    predictions_jsonl_path = output_root / "predictions.jsonl"
    submission_csv_path = output_root / "submission.csv"

    start_time = time.time()
    all_candidates: list[DetectionCandidate] = []
    image_summaries: list[dict[str, Any]] = []

    with raw_candidates_path.open("w", encoding="utf-8") as raw_handle:
        for image_index, image in enumerate(images, start=1):
            image_path = Path(images_dir).resolve() / image.file_name
            if not image_path.is_file():
                raise FileNotFoundError(f"Image does not exist: {image_path}")

            image_run_dir = image_runs_root / _safe_slug(Path(image.file_name).stem)
            image_run_dir.mkdir(parents=True, exist_ok=True)
            prompts_to_run = (PRIMARY_DETECTION_PROMPT,) + tuple(RESCUE_DETECTION_PROMPTS)

            per_prompt_candidates: list[DetectionCandidate] = []
            for prompt_index, prompt_text in enumerate(prompts_to_run, start=1):
                pass_dir = image_run_dir / f"proposal_pass_{prompt_index:02d}"
                pass_dir.mkdir(parents=True, exist_ok=True)
                history, final_outputs, rendered_image = runtime_lib.agent_inference(
                    img_path=str(image_path),
                    initial_text_prompt=prompt_text,
                    send_generate_request=send_request,
                    call_sam_service=local_service.call_service,
                    output_dir=str(pass_dir),
                    debug=bool(config.keep_debug_artifacts),
                    max_generations=int(config.max_generations),
                )
                if config.keep_debug_artifacts:
                    rendered_image.save(pass_dir / "final_selection.png")
                    with (pass_dir / "history.json").open("w", encoding="utf-8") as handle:
                        json.dump(history, handle, indent=2)
                        handle.write("\n")

                image_width = int(final_outputs.get("orig_img_w", image.width or 0))
                image_height = int(final_outputs.get("orig_img_h", image.height or 0))
                pred_boxes = list(final_outputs.get("pred_boxes", []))
                pred_scores = list(final_outputs.get("pred_scores", []))

                for proposal_index, bbox_xywh_norm in enumerate(pred_boxes):
                    abs_bbox = _denormalize_bbox_xywh(
                        bbox_xywh_norm,
                        image_width=image_width,
                        image_height=image_height,
                    )
                    proposal_score = 0.0
                    if proposal_index < len(pred_scores):
                        proposal_score = float(pred_scores[proposal_index])
                    candidate = DetectionCandidate(
                        image_id=image.id,
                        image_path=str(image_path),
                        source_prompt=prompt_text,
                        proposal_index=proposal_index,
                        bbox_xywh=abs_bbox,
                        proposal_score=proposal_score,
                    )
                    per_prompt_candidates.append(candidate)

            deduped_candidates = _dedupe_candidates(
                per_prompt_candidates,
                iou_threshold=float(config.proposal_iou_threshold),
            )
            classified_candidates: list[DetectionCandidate] = []

            for candidate_index, candidate in enumerate(deduped_candidates, start=1):
                candidate_dir = image_run_dir / f"candidate_{candidate_index:03d}"
                candidate_dir.mkdir(parents=True, exist_ok=True)
                overlay_path = candidate_dir / "overlay.jpg"
                crop_path = candidate_dir / "crop.jpg"
                _render_overlay_image(
                    image_path=candidate.image_path,
                    bbox_xywh=candidate.bbox_xywh,
                    output_path=str(overlay_path),
                )
                _save_crop_image(
                    image_path=candidate.image_path,
                    bbox_xywh=candidate.bbox_xywh,
                    context_ratio=float(config.crop_context_ratio),
                    output_path=str(crop_path),
                )
                candidate.overlay_path = str(overlay_path)
                candidate.crop_path = str(crop_path)

                classification = _classify_candidate(
                    send_request=send_request,
                    overlay_path=str(overlay_path),
                    crop_path=str(crop_path),
                    prompt_text=classification_prompt,
                    category_name_to_id=category_name_to_id,
                )
                candidate.category_id = classification["category_id"]
                candidate.category_name = classification["category_name"]
                candidate.classification_confidence = classification["confidence"]
                candidate.classification_reason = classification["reason"]
                candidate.drop_detection = classification["drop_detection"]
                candidate.final_score = (
                    float(candidate.proposal_score)
                    * float(candidate.classification_confidence)
                )

                raw_handle.write(json.dumps(_prediction_to_json(candidate)) + "\n")
                raw_handle.flush()

                if candidate.drop_detection:
                    continue
                if candidate.category_id is None:
                    continue
                if candidate.classification_confidence < float(
                    config.classification_min_confidence
                ):
                    continue

                classified_candidates.append(candidate)

            final_candidates = _final_nms(
                classified_candidates,
                iou_threshold=float(config.final_iou_threshold),
            )
            image_summaries.append(
                {
                    "image_index": image_index,
                    "image_id": image.id,
                    "file_name": image.file_name,
                    "proposal_count_raw": len(per_prompt_candidates),
                    "proposal_count_deduped": len(deduped_candidates),
                    "final_detection_count": len(final_candidates),
                }
            )
            all_candidates.extend(final_candidates)

    prediction_rows = [
        DetectionPrediction(
            image_id=candidate.image_id,
            category_id=int(candidate.category_id),
            bbox_xywh=candidate.bbox_xywh,
            score=float(candidate.final_score),
        )
        for candidate in all_candidates
        if candidate.category_id is not None
    ]
    write_submission_csv(prediction_rows, submission_csv_path)

    with predictions_jsonl_path.open("w", encoding="utf-8") as handle:
        for candidate in all_candidates:
            handle.write(json.dumps(_prediction_to_json(candidate)) + "\n")

    summary = {
        "competition": "fathomnet_2026_kaggle",
        "image_count": len(images),
        "raw_candidate_count": len(
            [summary["proposal_count_raw"] for summary in image_summaries]
        ),
        "final_detection_count": len(all_candidates),
        "submission_row_count": len(prediction_rows),
        "runtime_sec": round(time.time() - start_time, 3),
        "raw_candidates_path": str(raw_candidates_path),
        "predictions_jsonl_path": str(predictions_jsonl_path),
        "submission_csv_path": str(submission_csv_path),
        "image_summaries": image_summaries,
        "config": asdict(config),
    }
    summary["raw_candidate_count"] = int(
        sum(item["proposal_count_raw"] for item in image_summaries)
    )
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    return summary
