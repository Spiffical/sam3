#!/usr/bin/env python3
"""Run scene-adaptive text proposals plus temporal click recovery on fixed frames.

The first-pass runner produces one-frame SAM3-agent outputs for each fixed
frame. Before optional SAM3 text proposals, Sonnet can inspect the selected
frame and choose a short, visually appropriate phrase set from the benchmark's
evidence-backed candidates. Verified masks persist while temporal click mode
recovers visible life that text grounding misses. Whole-frame click review
remains off.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pycocotools import mask as mask_utils

import scripts.click_engine_probe as P
import scripts.tile_stray_bakeoff as B
from nibi_model_compare.frame_output_utils import decode_rle_to_mask
from nibi_model_compare.som_missed_creatures import (
    load_click_discovery_system_prompt,
    parse_creature_click_groups,
    render_existing_masks_overlay,
    render_grid_overlay,
    render_proposed_click_groups_overlay,
)
from scripts.click_engine_probe import (
    _iou,
    verify_clicks,
    verify_masks,
)
from scripts.probe_sam3_life_prompts import (
    PromptSpec,
    build_taxa_by_clip,
    git_state,
    production_prompt_bank,
    serialize_state,
    sha256,
)


DEFAULT_MODEL = "claude-fable-5"
DEFAULT_FIRSTPASS_MODEL = "claude-sonnet-4-6"
DEFAULT_STRATEGY = "S3_cons_temp"
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


def choose_mask_guided_pass_count(
    requested: int,
    *,
    initial_known_count: int,
    sparse_extra_pass: bool,
    finder_mode: str,
) -> int:
    if requested <= 0:
        return 0
    if (
        sparse_extra_pass
        and initial_known_count == 0
        and finder_mode in {"mask-guided", "hybrid"}
    ):
        return requested + 1
    return requested


def _first_positive_click(group: dict[str, Any]) -> dict[str, Any]:
    """Return the foreground seed; negative clicks only constrain that seed."""
    for click in group.get("clicks") or []:
        if int(click.get("label", -1)) == 1:
            return dict(click)
    raise ValueError("click group has no foreground seed")


def _filter_prior_attempt_groups(
    groups: list[dict[str, Any]],
    prior_attempts: list[dict[str, Any]],
    *,
    spatial_tolerance: float = 0.025,
    description_overlap_threshold: float = 0.60,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop only same-description, same-location cross-pass reattempts.

    This is not a click budget: a new identity, a materially different click,
    or an earlier failed identity with a genuinely corrected click still runs.
    """
    kept: list[dict[str, Any]] = []
    repeated: list[dict[str, Any]] = []
    for group in groups:
        positive = _first_positive_click(group)
        description_tokens = {
            token
            for token in re.findall(
                r"[a-z0-9]+", str(group.get("description", "")).lower()
            )
            if len(token) > 2
        }
        is_repeat = False
        for attempt in prior_attempts:
            click = attempt.get("click") or {}
            distance = float(np.hypot(
                float(positive.get("x", 0.0)) - float(click.get("x", 0.0)),
                float(positive.get("y", 0.0)) - float(click.get("y", 0.0)),
            ))
            if distance > spatial_tolerance:
                continue
            prior_tokens = {
                token
                for token in re.findall(
                    r"[a-z0-9]+",
                    str(attempt.get("description", "")).lower(),
                )
                if len(token) > 2
            }
            smaller = min(len(description_tokens), len(prior_tokens))
            overlap = (
                len(description_tokens.intersection(prior_tokens)) / smaller
                if smaller else 0.0
            )
            if overlap >= description_overlap_threshold:
                is_repeat = True
                break
        (repeated if is_repeat else kept).append(group)
    return kept, repeated


def _click_counts(clicks: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "positive": sum(int(click.get("label", -1)) == 1 for click in clicks),
        "negative": sum(int(click.get("label", -1)) == 0 for click in clicks),
    }


def _adaptive_prompt_planner_prompt(
    frame_id: str,
    visual_note: str,
    candidates: list[str],
) -> str:
    candidate_lines = "\n".join(f"- {phrase}" for phrase in candidates)
    return (
        "You are planning text queries for SAM3 on one underwater TARGET FRAME. "
        "Choose only phrases whose visual concept is credibly present and useful "
        "for proposing instance masks in this exact frame. Phrases are retrieval "
        "handles only, never taxonomic labels. Prefer the shortest complementary "
        "set; omit redundant, visually mismatched, or overly generic phrases. "
        "In particular, use 'small creatures' only when discrete small animal "
        "bodies are actually visible. Do not use it merely because a scene has "
        "fine coral branches, texture, or many sessile colonies. Dense coral, "
        "sea-fan, sea-whip, or sponge scenes should use the matching morphology "
        "or plausible provided taxon handle instead. On-screen logos, lettering, "
        "timestamps, and video/UI overlays are never targets. Click recovery will "
        "handle visible life that none of these phrases can ground, so it is valid "
        "to select no phrase.\n\n"
        f"Frame id: {frame_id}\n"
        f"Scene note: {visual_note or 'none'}\n"
        "Allowed candidates (copy selected strings exactly):\n"
        f"{candidate_lines or '- none'}\n\n"
        "Output brief reasoning followed by EXACTLY ONE trailing tag:\n"
        '<answer>{"phrases":["<exact allowed candidate>"]}</answer>'
    )


def _select_adaptive_prompt_specs(
    candidates: list[str], answer: dict[str, Any]
) -> list[PromptSpec]:
    allowed = {
        " ".join(candidate.split()).casefold(): " ".join(candidate.split())
        for candidate in candidates
        if " ".join(candidate.split())
    }
    selected: list[PromptSpec] = []
    seen: set[str] = set()
    raw_phrases = answer.get("phrases")
    if not isinstance(raw_phrases, list):
        return selected
    for value in raw_phrases:
        if not isinstance(value, str):
            continue
        key = " ".join(value.split()).casefold()
        if key in allowed and key not in seen:
            selected.append(
                PromptSpec(text=allowed[key], group="adaptive_scene")
            )
            seen.add(key)
    return selected


def _plan_adaptive_text_prompts(
    record: dict[str, Any],
    target_path: Path,
    output_dir: Path,
    *,
    model: str,
) -> tuple[list[PromptSpec], dict[str, Any]]:
    candidates = [
        str(value).strip()
        for value in record.get("sam3_proposal_prompts") or []
        if str(value).strip()
    ]
    prompt = _adaptive_prompt_planner_prompt(
        str(record.get("id", "unknown")),
        str(record.get("visual_note", "")),
        candidates,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    response = P.send_claude_request(
        [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(target_path)},
                {"type": "text", "text": prompt},
            ],
        }],
        model=model,
        max_tokens=P.response_token_budget(model, 800),
    )
    (output_dir / "adaptive_prompt_plan.txt").write_text(
        response or "<none>", encoding="utf-8"
    )
    answer = P._extract_answer_json(response) or {}
    selected = _select_adaptive_prompt_specs(candidates, answer)
    metadata = {
        "planner": model,
        "candidate_phrases": candidates,
        "selected_phrases": [spec.text for spec in selected],
        "parse_failed": not isinstance(answer.get("phrases"), list),
        "api_failed": response is None,
    }
    _write_json(output_dir / "adaptive_prompt_plan.json", metadata)
    return selected, metadata


def _drop_conflicting_negative_clicks(
    groups: list[dict[str, Any]],
    width: int,
    height: int,
    *,
    min_distance_px: float = 6.0,
) -> tuple[list[dict[str, Any]], int]:
    """Drop exclude clicks that contradict a localized include click.

    Discovery negatives are expressed before the foreground point is precisely
    relocated. If relocation puts the positive and a retained negative on the
    same pixels, sending both labels to SAM3 is undefined and cannot express the
    model's intended constraint.
    """
    cleaned: list[dict[str, Any]] = []
    removed = 0
    for group in groups:
        clicks = [dict(click) for click in group.get("clicks") or []]
        positives = [click for click in clicks if click.get("label") == 1]
        kept_clicks: list[dict[str, Any]] = []
        for click in clicks:
            if click.get("label") != 0:
                kept_clicks.append(click)
                continue
            conflicts = any(
                np.hypot(
                    (float(click.get("x", 0.0)) - float(pos.get("x", 0.0)))
                    * width,
                    (float(click.get("y", 0.0)) - float(pos.get("y", 0.0)))
                    * height,
                )
                < min_distance_px
                for pos in positives
            )
            if conflicts:
                removed += 1
            else:
                kept_clicks.append(click)
        cleaned.append({**group, "clicks": kept_clicks})
    return cleaned, removed


def _mask_guided_discovery_prompt(
    *,
    pass_instruction: str,
    has_focus_crops: bool = False,
) -> str:
    """Recall-first all-life discovery contract with explicit SAM3 negatives."""
    focus_description = (
        "The FIFTH image is an enlarged untouched crop of the required focus "
        "region. The SIXTH image is the matching enlarged strong-mask crop, "
        "retaining the full-frame coordinate grid. All returned coordinates "
        "must remain normalized to the FULL FRAME, not the crop. "
        if has_focus_crops else ""
    )
    return (
        "The first image is the untouched raw TARGET FRAME. The second image is "
        "the same target with lightly translucent green regions showing life "
        "masks already accepted from earlier stages; its 10x10 grid gives "
        "normalized coordinate context. The THIRD image is a strong green "
        "coverage map of the same accepted masks. Use the strong map to decide "
        "whether a particular structure is already covered, and use the raw and "
        "light views to inspect its biological boundaries. Compare the raw and "
        "masked target pixel-for-pixel: faint life beside or behind a green region is still "
        "missed when its own structure is not green. Do not assume a cluster is "
        "fully covered merely because nearby organisms are green. The FOURTH "
        "image is an outline-only map of accepted masks: green contours show "
        "their outer boundaries while leaving the raw interior visible. This is "
        "the decisive view for crowded branching scenes. A separate coral, whip, "
        "or colony visible through the filled inter-branch space of an accepted "
        "outer silhouette is NOT thereby segmented; report it if it has its own "
        "visually separable branching system or depth layer. "
        + focus_description
        + "Subsequent images are raw REFERENCE "
        "FRAMES from nearby times. "
        + pass_instruction
        + "Find every real, visually detectable marine organism or coherent "
        "colony in the TARGET FRAME that is NOT adequately covered by green. "
        "Targets include mobile animals; corals, sea fans, sea whips, sponges, "
        "anemones, hydroids and other sessile or colonial animals; and coherent "
        "macroalgae, seagrass, or other plant-like marine life. Treat each "
        "visually separable organism or coherent colony as a target. Include "
        "small, camouflaged, stationary, and frame-clipped life when credible "
        "biological structure or temporal support is visible. Do not click "
        "green-covered life, marine snow, bare substrate, debris, shell "
        "fragments, shadows, unsupported blobs, or any on-screen logo, text, "
        "timestamp, grid line, or other video/UI overlay. If an organism passes "
        "behind an overlay, click only a visible biological part outside it. "
        "Return an empty list only "
        "after an exhaustive scan finds no supported missed life.\n\n"
        "Return one small click GROUP per missed target. Every group must start "
        "with label 1 on a safe interior part of that target. Label 0 means "
        "EXCLUDE THIS LOCATION FROM THIS TARGET'S MASK; it does not mean that "
        "the location contains no life. Add one or two label-0 clicks when the "
        "target touches substrate, a green mask, or a visually separable "
        "neighbour that SAM3 may merge. Put each negative inside the unwanted "
        "region, never on another desired part of the same organism. High-value "
        "negative locations include a touching but visually separable coral or "
        "plant, a substrate lobe protruding beyond the target's outer silhouette, "
        "or a disconnected spill. Do not use a negative click solely to carve "
        "small spaces between fine branches: filling those enclosed spaces is "
        "acceptable when the mask still follows one complete object's silhouette. "
        "ONE GROUP MUST REPRESENT EXACTLY ONE COMPLETE BIOLOGICAL IDENTITY. "
        "Never redefine a branch, appendage, small visible patch, or other "
        "subregion of a larger continuous organism/colony as a separate target; "
        "place positives across the complete visible extent instead. Conversely, "
        "never put positives from visually separable foreground and background "
        "organisms/colonies into one group, even if they overlap, touch, look "
        "similar, or share a taxon. A depth-layer change, occlusion boundary, or "
        "separate branching system requires separate groups. Use a "
        "second label-1 click only for another part of the same elongated or "
        "fragmented target. Otherwise, one central positive click is best.\n\n"
        "Coordinate accuracy is critical. Before submitting, verify that every "
        "positive coordinate visibly lands ON the described target in the grid, "
        "not in nearby water or between branches. For branching life, prefer a "
        "thick, high-contrast branch or solid base that is safely inside the "
        "target.\n\n"
        "Output free-text reasoning followed by EXACTLY ONE trailing tag:\n"
        '<answer>{"missed_creatures":[{"id":1,"description":"<short>",'
        '"clicks":[{"x":<float>,"y":<float>,"label":1},'
        '{"x":<float>,"y":<float>,"label":0}]}]}</answer>'
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="configs/presentation_benchmark_frames.json",
    )
    parser.add_argument(
        "--firstpass-root",
        required=True,
        help="Directory containing <frame-id>/frame_outputs_rle.json and summary.json",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--phrase-model",
        default="",
        help=(
            "Model used only to select adaptive SAM3 text phrases. Empty "
            "uses --model. Proposal verification and all click reasoning "
            "continue to use --model."
        ),
    )
    parser.add_argument(
        "--effort",
        choices=("", "low", "medium", "high", "xhigh", "max"),
        default="",
        help=(
            "Anthropic adaptive-thinking effort. Medium is recommended for "
            "Sonnet 5 visual discovery/refinement; empty uses the API default."
        ),
    )
    parser.add_argument(
        "--firstpass-model",
        default=DEFAULT_FIRSTPASS_MODEL,
        help="Expected model recorded by the reused first-pass run",
    )
    parser.add_argument(
        "--initial-mask-root",
        default="",
        help=(
            "Optional prior custom-flow root containing "
            "<frame-id>/final_masks_rle.json. These accepted masks seed the new "
            "run so residual discovery can continue without recomputing them."
        ),
    )
    parser.add_argument(
        "--skip-firstpass",
        action="store_true",
        help=(
            "Do not reload first-pass masks. Use this when --initial-mask-root "
            "already contains the complete accepted mask set and the run is "
            "only continuing residual discovery/refinement."
        ),
    )
    parser.add_argument(
        "--selection-manifest",
        default="",
        help=(
            "SeaTube selection manifest containing per-clip annotations. "
            "Defaults to the benchmark manifest's selection_manifest field."
        ),
    )
    parser.add_argument(
        "--text-proposal-mode",
        choices=("none", "adaptive", "compact", "broad"),
        default="adaptive",
        help=(
            "SAM3 text proposals before click recovery. Adaptive lets the MLLM "
            "select visually appropriate manifest phrases; compact is the "
            "taxon-first production bank; broad is diagnostic and slower."
        ),
    )
    parser.add_argument(
        "--text-proposal-threshold",
        type=float,
        default=0.40,
        help="Minimum SAM3 score for a text proposal (default 0.40)",
    )
    parser.add_argument(
        "--min-text-confidence",
        type=float,
        default=0.50,
        help="Minimum MLLM post-mask confidence for a text proposal",
    )
    parser.add_argument(
        "--finder-mode",
        choices=("mask-guided", "hybrid", "s3"),
        default="mask-guided",
        help=(
            "Candidate finder: persistent first-pass mask guidance (default), "
            "mask guidance plus the cold S3 finder, or cold S3 only."
        ),
    )
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY,
                        choices=sorted(B.STRATEGIES))
    parser.add_argument(
        "--frame-id",
        action="append",
        default=[],
        help="Run only this manifest frame id; repeat for multiple ids",
    )
    parser.add_argument(
        "--temporal-offsets",
        default="15,30,45",
        help="Raw-video frame offsets for temporal context",
    )
    parser.add_argument(
        "--temporal-seconds",
        default="",
        help=(
            "Optional comma-separated temporal offsets in seconds. When set, "
            "these are converted per video using its decoded FPS and override "
            "--temporal-offsets."
        ),
    )
    parser.add_argument(
        "--max-clicks",
        type=int,
        default=0,
        help=(
            "Maximum clicks per object; 0 (default) is unlimited and lets the "
            "MLLM decide when the mask is good, rejected, or abandoned."
        ),
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--mask-generator",
        choices=("hybrid", "full", "zoom"),
        default="hybrid",
        help="SAM3 click-mask generator (default: hybrid)",
    )
    parser.add_argument(
        "--max-click-groups-per-pass",
        type=int,
        default=0,
        help=(
            "Diagnostic cap on discovered targets refined per pass; 0 keeps all. "
            "Do not use a positive cap for final recall runs."
        ),
    )
    parser.add_argument(
        "--zoom-crop-frac",
        type=float,
        default=0.50,
        help=(
            "Full-frame width/height fraction used by the zoom mask generator "
            "(default 0.50, large enough for branching colonies)."
        ),
    )
    parser.add_argument(
        "--click-localization-crop-frac",
        type=float,
        default=0.30,
        help=(
            "Full-frame width/height fraction shown when Sonnet relocates an "
            "approximate discovery click (default 0.30)."
        ),
    )
    parser.add_argument(
        "--visual-qa-focus",
        default="",
        help=(
            "Optional reviewer-provided residual region/object description to "
            "prioritize during discovery. Sonnet still chooses all clicks."
        ),
    )
    parser.add_argument(
        "--mask-guided-passes",
        type=int,
        default=1,
        help=(
            "Sequential persistent-mask-guided discovery/refinement passes "
            "(default 1). Use 0 to continue until a complete pass adds no new "
            "accepted masks. Each pass sees masks accepted earlier."
        ),
    )
    parser.add_argument(
        "--border-scan",
        choices=("off", "last", "every"),
        default="off",
        help=(
            "When to append the expensive high-resolution perimeter scan to "
            "ordinary discovery (default: off)."
        ),
    )
    parser.add_argument(
        "--sparse-extra-pass",
        action="store_true",
        help=(
            "Add one mask-guided discovery pass when first-pass plus verified "
            "text proposals leave the frame with zero accepted masks."
        ),
    )
    parser.add_argument(
        "--min-recovery-confidence",
        type=float,
        default=0.70,
        help=(
            "Minimum post-mask creature confidence for a genuinely new mask. "
            "Re-finds of first-pass masks are retained regardless (default 0.70)."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def _encode_mask(mask: np.ndarray) -> dict[str, Any]:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return {"size": [int(rle["size"][0]), int(rle["size"][1])],
            "counts": counts}


def _load_firstpass(frame_dir: Path, expected_model: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary = _read_json(frame_dir / "summary.json")
    actual_model = str(summary.get("model", ""))
    if actual_model != expected_model:
        raise RuntimeError(
            f"first-pass model mismatch in {frame_dir}: "
            f"expected {expected_model!r}, found {actual_model!r}"
        )
    if int(summary.get("error_count", 0)) != 0:
        raise RuntimeError(f"first-pass run has errors: {frame_dir}")

    doc = _read_json(frame_dir / "frame_outputs_rle.json")
    if len(doc.get("frames", [])) != 1:
        raise RuntimeError(f"expected exactly one first-pass frame: {frame_dir}")
    height, width = [int(v) for v in doc["frame_size_hw"]]
    record = doc["frames"][0]
    probs = list(record.get("out_probs") or [])
    boxes = list(record.get("out_boxes_xywh") or [])
    masks = []
    for index, rle in enumerate(record.get("out_binary_masks_rle") or []):
        mask = decode_rle_to_mask(rle, height, width).astype(bool)
        masks.append({
            "mask": mask,
            "prob": float(probs[index]) if index < len(probs) else None,
            "box_xywh": boxes[index] if index < len(boxes) else None,
            "source": "firstpass",
        })
    return masks, summary


def _load_initial_masks(
    root: Path | None,
    frame_id: str,
    expected_shape: tuple[int, int],
) -> list[dict[str, Any]]:
    """Load accepted masks from a prior custom-flow run for residual discovery."""
    if root is None:
        return []
    path = root / frame_id / "final_masks_rle.json"
    if not path.exists():
        path = root / frame_id / "checkpoint_masks_rle.json"
    doc = _read_json(path)
    height, width = [int(value) for value in doc["frame_size_hw"]]
    if (height, width) != expected_shape:
        raise RuntimeError(
            f"initial mask/frame size mismatch for {frame_id}: "
            f"{(height, width)} vs {expected_shape}"
        )
    return [
        {
            "mask": decode_rle_to_mask(rle, height, width).astype(bool),
            "source": "initial",
        }
        for rle in doc.get("masks", [])
    ]


def _should_run_border_scan(
    mode: str,
    *,
    convergence_mode: bool,
    pass_index: int,
    requested_passes: int,
) -> bool:
    """Schedule edge review without preventing agent-controlled convergence.

    In a fixed sweep, ``every`` and ``last`` retain their literal meanings. In
    convergence mode there is no knowable last pass, so either enabled mode
    performs one dedicated edge audit on pass 1. Subsequent full-frame passes
    remain unlimited and can eventually return the explicit empty stop signal.
    """
    if mode == "off":
        return False
    if convergence_mode:
        return pass_index == 1
    if mode == "every":
        return True
    return mode == "last" and pass_index == requested_passes


def _discovery_focus_region(
    pass_index: int,
    pass_count: int,
) -> tuple[float, float, float, float, str]:
    """Return normalized focus bounds for an exhaustive spatial sweep."""
    if pass_count == 0:
        # Unlimited convergence still needs a deterministic minimum-coverage
        # scan before the model is allowed to stop.  Crowded scenes are easy to
        # dismiss from one full-frame view, so use four 2x2 tiles first, then
        # let subsequent passes review the full residual frame until Sonnet
        # explicitly reports that no missed life remains.
        if pass_index <= 4:
            return _discovery_focus_region(pass_index, 4)
        return 0.0, 1.0, 0.0, 1.0, "the entire residual frame"
    if pass_count <= 1:
        return 0.0, 1.0, 0.0, 1.0, "the entire frame"
    if pass_count <= 3:
        left = (pass_index - 1) / pass_count
        right = pass_index / pass_count
        label = (
            ("left half", "right half")[pass_index - 1]
            if pass_count == 2
            else ("left third", "middle third", "right third")[pass_index - 1]
        )
        return left, right, 0.0, 1.0, label
    columns = int(np.ceil(np.sqrt(pass_count)))
    rows = int(np.ceil(pass_count / columns))
    zero_based = pass_index - 1
    row = zero_based // columns
    column = zero_based % columns
    left = column / columns
    right = min(1.0, (column + 1) / columns)
    top = row / rows
    bottom = min(1.0, (row + 1) / rows)
    return (
        left,
        right,
        top,
        bottom,
        f"grid tile x={left:.2f}..{right:.2f}, y={top:.2f}..{bottom:.2f}",
    )


def _read_raw_frame(video_path: Path, frame_index: int) -> tuple[np.ndarray, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"could not read frame {frame_index} from {video_path}")
    return frame, fps


def _mask_level_nms(
    results: list[dict[str, Any]],
    threshold: float = 0.5,
    containment_threshold: float = 0.9,
) -> tuple[list[dict[str, Any]], int]:
    """Remove duplicate or broad containing masks, preferring tighter masks.

    IoU alone misses a common crowded-coral failure: a clean individual mask is
    wholly contained by a much larger mask that merged a neighbouring colony.
    Processing smaller masks first lets strict containment reject that broad
    container only when the broad proposal's own seed is also inside the tight
    mask. Distinct seeds preserve interlaced colonies whose filled silhouettes
    legitimately overlap.
    """
    def seed_inside(result: dict[str, Any], mask: np.ndarray) -> bool:
        click = result.get("seed_click") or {}
        if not isinstance(click.get("x"), (int, float)) or not isinstance(
            click.get("y"), (int, float)
        ):
            return False
        height, width = mask.shape
        x = min(width - 1, max(0, int(round(float(click["x"]) * (width - 1)))))
        y = min(height - 1, max(0, int(round(float(click["y"]) * (height - 1)))))
        return bool(mask[y, x])

    kept: list[tuple[int, dict[str, Any]]] = []
    removed = 0
    order = sorted(
        range(len(results)),
        key=lambda index: int(np.asarray(results[index]["mask"]).sum()),
    )
    for index in order:
        result = results[index]
        mask = np.asarray(result["mask"]).astype(bool)
        if mask.any() and any(
            _iou(mask, np.asarray(old["mask"]).astype(bool)) >= threshold
            or (
                _overlap_coefficient(mask, old["mask"])
                >= containment_threshold
                and seed_inside(
                    result, np.asarray(old["mask"]).astype(bool)
                )
            )
            for _old_index, old in kept
        ):
            removed += 1
            continue
        kept.append((index, result))
    kept.sort(key=lambda item: item[0])
    return [result for _index, result in kept], removed


def _overlap_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    smaller = min(int(a.sum()), int(b.sum()))
    if smaller == 0:
        return 0.0
    return float(np.logical_and(a, b).sum()) / float(smaller)


def _ambiguous_overlap_components(
    masks: list[np.ndarray],
    *,
    containment_threshold: float = 0.40,
    max_gap_fraction: float = 0.006,
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    """Find overlapping or near-contact masks that may share one identity.

    Geometry only selects candidates for identity review; it never merges them.
    Temporal/visual identity evidence and a strict union-mask verifier make the
    eventual consolidation decision.
    """
    adjacency = {index: set() for index in range(len(masks))}
    pairs: list[dict[str, Any]] = []
    normalized_masks = [np.asarray(mask).astype(bool) for mask in masks]
    if normalized_masks:
        height, width = normalized_masks[0].shape
        max_gap_px = float(max_gap_fraction) * float(np.hypot(width, height))
        distance_maps = [
            cv2.distanceTransform(
                np.logical_not(mask).astype(np.uint8), cv2.DIST_L2, 3
            )
            for mask in normalized_masks
        ]
    else:
        max_gap_px = 0.0
        distance_maps = []
    for left in range(len(masks)):
        for right in range(left + 1, len(masks)):
            overlap = _overlap_coefficient(masks[left], masks[right])
            gap_px = float(min(
                distance_maps[left][normalized_masks[right]].min(),
                distance_maps[right][normalized_masks[left]].min(),
            ))
            trigger = (
                "overlap" if overlap >= containment_threshold
                else "near_contact" if gap_px <= max_gap_px
                else None
            )
            if trigger is None:
                continue
            iou = _iou(
                np.asarray(masks[left]).astype(bool),
                np.asarray(masks[right]).astype(bool),
            )
            adjacency[left].add(right)
            adjacency[right].add(left)
            pairs.append(
                {
                    "left_index": left,
                    "right_index": right,
                    "smaller_mask_overlap": float(overlap),
                    "iou": float(iou),
                    "gap_px": gap_px,
                    "trigger": trigger,
                }
            )
    components: list[list[int]] = []
    unseen = {index for index, neighbours in adjacency.items() if neighbours}
    while unseen:
        start = min(unseen)
        stack = [start]
        component: set[int] = set()
        while stack:
            index = stack.pop()
            if index in component:
                continue
            component.add(index)
            stack.extend(adjacency[index] - component)
        unseen -= component
        components.append(sorted(component))
    return components, pairs


def _duplicates_known_mask(
    mask: np.ndarray,
    known: list[dict[str, Any]],
    *,
    iou_threshold: float = 0.5,
    containment_threshold: float = 0.8,
) -> bool:
    """Reject repeat segmentations without merging nearby distinct organisms."""
    return any(
        _iou(mask, np.asarray(item["mask"]).astype(bool)) >= iou_threshold
        or _overlap_coefficient(mask, item["mask"]) >= containment_threshold
        for item in known
    )


def _propose_text_masks(
    service: Any,
    target_path: Path,
    specs: list[PromptSpec],
    known_masks: list[dict[str, Any]],
    *,
    threshold: float,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run many SAM3 text prompts while encoding the target image only once."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if not specs:
        metadata = {
            "threshold": threshold,
            "n_prompts": 0,
            "n_candidates_before_mllm_verify": 0,
            "prompt_rows": [],
        }
        _write_json(output_dir / "proposal_summary.json", metadata)
        return [], metadata

    from PIL import Image
    from sam3.agent.client_sam3 import remove_overlapping_masks

    processor = service.processor
    previous_threshold = float(processor.confidence_threshold)
    accepted: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    candidate_id = 0
    image = Image.open(target_path).convert("RGB")
    state = None
    try:
        processor.set_confidence_threshold(threshold)
        state = processor.set_image(image)
        for prompt_index, spec in enumerate(specs, 1):
            processor.reset_all_prompts(state)
            state = processor.set_text_prompt(prompt=spec.text, state=state)
            outputs = remove_overlapping_masks(
                serialize_state(state, image.width, image.height)
            )
            order = sorted(
                range(len(outputs.get("pred_scores", []))),
                key=lambda index: float(outputs["pred_scores"][index]),
                reverse=True,
            )
            kept_for_prompt = 0
            duplicate_count = 0
            prompt_scores: list[float] = []
            for output_index in order:
                score = float(outputs["pred_scores"][output_index])
                if score < threshold:
                    continue
                mask = decode_rle_to_mask(
                    outputs["pred_masks"][output_index], image.height, image.width
                ).astype(bool)
                if not mask.any():
                    continue
                if _duplicates_known_mask(
                    mask, [*known_masks, *accepted]
                ):
                    duplicate_count += 1
                    continue
                candidate_id += 1
                anchor_x, anchor_y = _mask_anchor(mask)
                accepted.append(
                    {
                        "creature_id": candidate_id,
                        "description": f"SAM3 text prompt: {spec.text}",
                        "mask": mask,
                        "score": score,
                        "sam3_score": score,
                        "box_xywh": (
                            outputs.get("pred_boxes")
                            or [None] * len(outputs.get("pred_scores", []))
                        )[output_index],
                        "source": "T",
                        "source_prompt": spec.text,
                        "source_group": spec.group,
                        "source_taxon": spec.source_taxon,
                        "seed_click": {
                            "x": anchor_x / max(1, image.width - 1),
                            "y": anchor_y / max(1, image.height - 1),
                            "label": 1,
                        },
                    }
                )
                kept_for_prompt += 1
                prompt_scores.append(score)
            prompt_rows.append(
                {
                    "prompt_index": prompt_index,
                    "prompt": spec.text,
                    "group": spec.group,
                    "source_taxon": spec.source_taxon,
                    "n_sam3_after_within_prompt_overlap_removal": len(order),
                    "n_kept_before_mllm_verify": kept_for_prompt,
                    "n_duplicate_known_or_prior_prompt": duplicate_count,
                    "kept_scores": prompt_scores,
                }
            )
    finally:
        processor.set_confidence_threshold(previous_threshold)

    metadata = {
        "threshold": threshold,
        "n_prompts": len(specs),
        "n_candidates_before_mllm_verify": len(accepted),
        "prompt_rows": prompt_rows,
    }
    _write_json(output_dir / "proposal_summary.json", metadata)
    return accepted, metadata


def _match_firstpass(
    click_results: list[dict[str, Any]],
    firstpass: list[dict[str, Any]],
    threshold: float = 0.5,
) -> dict[int, dict[str, Any]]:
    pairs = []
    for click_index, result in enumerate(click_results):
        mask = np.asarray(result["mask"]).astype(bool)
        if not mask.any():
            continue
        for first_index, existing in enumerate(firstpass):
            pairs.append((_iou(mask, existing["mask"]), click_index, first_index))
    used_clicks: set[int] = set()
    used_firstpass: set[int] = set()
    matches: dict[int, dict[str, Any]] = {}
    for iou, click_index, first_index in sorted(pairs, reverse=True):
        if iou < threshold or click_index in used_clicks or first_index in used_firstpass:
            continue
        used_clicks.add(click_index)
        used_firstpass.add(first_index)
        matches[click_index] = {
            "firstpass_index": first_index,
            "iou": float(iou),
            "method": "iou",
        }

    # A click-mode refiner can return a high-quality *part* of a creature (for
    # example only a fish's head/fins).  Its union IoU with the full first-pass
    # mask can then fall below 0.5 even though the new mask is almost entirely a
    # subset of that known creature.  Treat it as a re-find when either the seed
    # click is inside an existing mask or >=80% of the smaller mask overlaps.
    # This is safer than lowering the global IoU threshold, which could merge
    # nearby distinct animals.
    for click_index, result in enumerate(click_results):
        if click_index in matches:
            continue
        click_mask = np.asarray(result["mask"]).astype(bool)
        if not click_mask.any():
            continue
        click = result.get("seed_click") or {}
        px = int(round(float(click.get("x", -1.0)) * (click_mask.shape[1] - 1)))
        py = int(round(float(click.get("y", -1.0)) * (click_mask.shape[0] - 1)))
        candidates = []
        for first_index, existing in enumerate(firstpass):
            known = existing["mask"]
            intersection = int(np.logical_and(click_mask, known).sum())
            union = int(np.logical_or(click_mask, known).sum())
            iou = intersection / union if union else 0.0
            smaller = min(int(click_mask.sum()), int(known.sum()))
            overlap = intersection / smaller if smaller else 0.0
            seed_inside = (
                0 <= px < known.shape[1]
                and 0 <= py < known.shape[0]
                and bool(known[py, px])
            )
            if seed_inside or overlap >= 0.8:
                method = "seed_inside_firstpass" if seed_inside else "mask_containment"
                candidates.append((seed_inside, overlap, iou, first_index, method))
        if candidates:
            _inside, overlap, iou, first_index, method = max(candidates)
            matches[click_index] = {
                "firstpass_index": first_index,
                "iou": float(iou),
                "overlap_coefficient": float(overlap),
                "method": method,
            }
    return matches


def _mask_anchor(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 8, 24
    return int(np.median(xs)), int(np.median(ys))


def _replacement_for_contained_mask(
    result: dict[str, Any],
    match: dict[str, Any],
    known_masks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Promote a verified, substantially more complete version of one mask.

    This is intentionally narrower than ordinary re-find matching: the old mask
    must be almost entirely contained in the new one, the expansion must be
    plausible rather than enormous, and Sonnet's post-mask verifier must be
    confident.  It lets a complete colony replace an earlier fragment without
    adding a duplicate instance.
    """
    if str(match.get("method")) != "mask_containment":
        return None
    known_index = int(match["firstpass_index"])
    if not 0 <= known_index < len(known_masks):
        return None
    confidence = float(result.get("creature_confidence", 0.0))
    containment = float(match.get("overlap_coefficient", 0.0))
    new_area = int(np.asarray(result["mask"]).astype(bool).sum())
    old_area = int(
        np.asarray(known_masks[known_index]["mask"]).astype(bool).sum()
    )
    ratio = new_area / old_area if old_area else 0.0
    if confidence < 0.75 or containment < 0.95:
        return None
    if not 1.5 <= ratio <= 8.0:
        return None
    replacement = dict(result)
    replacement["source"] = str(
        known_masks[known_index].get("source", "replacement")
    )
    replacement["replacement_area_ratio"] = ratio
    return {
        "known_index": known_index,
        "result": replacement,
        "old_area_px": old_area,
        "new_area_px": new_area,
        "area_ratio": ratio,
        "containment": containment,
        "confidence": confidence,
    }


def _draw_numbered_masks(frame: np.ndarray, masks: list[np.ndarray], path: Path) -> None:
    out = frame.copy()
    for index, mask in enumerate(masks, 1):
        color = COLORS_BGR[(index - 1) % len(COLORS_BGR)]
        overlay = out.copy()
        overlay[mask] = color
        out = cv2.addWeighted(overlay, 0.30, out, 0.70, 0)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2, cv2.LINE_AA)
        x, y = _mask_anchor(mask)
        label = str(index)
        cv2.putText(out, label, (x - 6, y + 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, label, (x - 6, y + 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), out)


def _draw_diagnostic(
    frame: np.ndarray,
    firstpass: list[dict[str, Any]],
    click_results: list[dict[str, Any]],
    matches: dict[int, tuple[int, float]],
    dropped: list[dict[str, Any]],
    path: Path,
) -> None:
    out = frame.copy()
    for existing in firstpass:
        contours, _ = cv2.findContours(existing["mask"].astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 220, 0), 2, cv2.LINE_AA)
    for index, result in enumerate(click_results):
        mask = np.asarray(result["mask"]).astype(bool)
        if not mask.any():
            continue
        matched = index in matches
        color = (0, 210, 210) if matched else (255, 255, 0)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2, cv2.LINE_AA)
        click = result.get("seed_click") or {}
        if "x" in click and "y" in click:
            point = (int(float(click["x"]) * out.shape[1]),
                     int(float(click["y"]) * out.shape[0]))
            cv2.drawMarker(out, point, color, cv2.MARKER_TILTED_CROSS, 20, 2)
    for result in dropped:
        mask = np.asarray(result["mask"]).astype(bool)
        if not mask.any():
            continue
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 0, 200), 1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    legend = (f"first-pass green={len(firstpass)}  re-found yellow={len(matches)}  "
              f"new cyan={len(click_results) - len(matches)}  dropped red={len(dropped)}")
    cv2.putText(out, legend, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), out)


def _draw_source_diagnostic(
    frame: np.ndarray,
    existing_masks: list[dict[str, Any]],
    text_masks: list[dict[str, Any]],
    click_masks: list[dict[str, Any]],
    path: Path,
    *,
    click_model: str,
) -> None:
    """Show which pipeline stage contributed each final persistent mask."""
    out = frame.copy()
    layers = (
        (existing_masks, (0, 220, 0)),  # green: prior accepted masks
        (text_masks, (0, 150, 255)),    # orange: SAM3 text proposal
        (click_masks, (255, 255, 0)),   # cyan: current click recovery
    )
    for items, color in layers:
        for item in items:
            mask = np.asarray(item["mask"]).astype(bool)
            if not mask.any():
                continue
            overlay = out.copy()
            overlay[mask] = color
            out = cv2.addWeighted(overlay, 0.22, out, 0.78, 0)
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(out, contours, -1, color, 2, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
    model_label = click_model.removeprefix("claude-")
    legend = (
        f"existing green={len(existing_masks)}  "
        f"SAM3 text orange={len(text_masks)}  "
        f"{model_label} click cyan={len(click_masks)}"
    )
    cv2.putText(
        out, legend, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
        (255, 255, 255), 1, cv2.LINE_AA
    )
    cv2.imwrite(str(path), out)


def _api_failed(frame_dir: Path) -> bool:
    for path in frame_dir.rglob("*.txt"):
        try:
            if "<none>" in path.read_text(encoding="utf-8", errors="replace"):
                return True
        except OSError:
            continue
    return False


def _label_crop(crop: np.ndarray, label: str) -> np.ndarray:
    out = crop.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(
        out,
        label,
        (6, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def _find_border_groups(
    frame: np.ndarray,
    overlay: np.ndarray,
    *,
    raw_index: int,
    temporal_offsets: list[int],
    guided_dir: Path,
    model: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """High-resolution scan for animals clipped by an image boundary."""
    height, width = frame.shape[:2]
    x_span = max(1, int(round(width * 0.30)))
    y_span = max(1, int(round(height * 0.30)))
    boxes = {
        "left": (0, 0, x_span, height),
        "right": (width - x_span, 0, width, height),
        "top": (0, 0, width, y_span),
        "bottom": (0, height - y_span, width, height),
    }
    neighbour = None
    neighbour_offset = None
    for offset in temporal_offsets:
        for signed in (offset, -offset):
            if raw_index + signed < 0:
                continue
            try:
                neighbour = P.read_video_frame(raw_index + signed)
                neighbour_offset = signed
                break
            except SystemExit:
                continue
        if neighbour is not None:
            break
    if neighbour is None:
        neighbour = frame
        neighbour_offset = 0

    paths: list[tuple[str, Path, Path]] = []
    for edge, (left, top, right, bottom) in boxes.items():
        target_crop = _label_crop(
            overlay[top:bottom, left:right],
            f"TARGET {edge.upper()} EDGE (green = already found)",
        )
        neighbour_crop = _label_crop(
            neighbour[top:bottom, left:right],
            f"REFERENCE {edge.upper()} EDGE ({neighbour_offset:+d} frames)",
        )
        target_path = guided_dir / f"border_{edge}_target.png"
        neighbour_path = guided_dir / f"border_{edge}_reference.png"
        cv2.imwrite(str(target_path), target_crop)
        cv2.imwrite(str(neighbour_path), neighbour_crop)
        paths.append((edge, target_path, neighbour_path))

    user_text = (
        "This is a high-resolution perimeter scan of one underwater TARGET "
        "frame. Images come in TARGET/REFERENCE pairs for the LEFT, RIGHT, TOP, "
        "and BOTTOM edge. Green regions in TARGET crops are marine life already "
        "segmented. Find only real organisms that are visibly clipped by an image "
        "boundary and are not green-covered. Partial organisms count even when "
        "only a body sliver or biological structure is inside the frame, but do not report "
        "marine snow, glare, substrate, or a merely blurry blob. Coherent pale "
        "white/gray branching coral, sea-fan, or sea-whip structures are life, "
        "not debris. On-screen logos, lettering, timestamps, and other video/UI "
        "overlays are never life; do not click them. Green coverage is exact: "
        "an ungreen branch is still missed "
        "when it touches or passes beside a green-covered colony. Use the paired "
        "time as supporting evidence. Coordinates must be normalized within the "
        "named EDGE CROP, not the full frame. Return an empty list if there is no "
        "confident unmasked boundary organism. Every target must begin with a "
        "label-1 click on the organism. Label 0 means exclude that location from "
        "this target's mask: add it inside adjacent substrate, green-covered life, "
        "or a touching neighbour when SAM3 might merge them. A negative click is "
        "not a standalone target and must never land on another desired part of "
        "the same organism.\n\n"
        "Output free-text reasoning followed by EXACTLY ONE trailing tag:\n"
        '<answer>{"missed_creatures":[{"edge":"left|right|top|bottom",'
        '"description":"<short>","clicks":[{"x":<float>,"y":<float>,'
        '"label":1},{"x":<float>,"y":<float>,"label":0}]}]}</answer>'
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for _edge, target_path, neighbour_path in paths:
        content.append({"type": "image", "image": str(target_path)})
        content.append({"type": "image", "image": str(neighbour_path)})
    response = P.send_claude_request(
        [{"role": "user", "content": content}],
        model=model,
        max_tokens=P.response_token_budget(model, 1200),
        # This is an optional fallback, not the primary discovery path. A
        # missing response should be recorded without multiplying latency.
        max_retries=1,
    )
    (guided_dir / "border_response.txt").write_text(
        response or "<none>", encoding="utf-8"
    )
    answer = P._extract_answer_json(response) or {}
    raw_groups = answer.get("missed_creatures")
    groups: list[dict[str, Any]] = []
    if isinstance(raw_groups, list):
        for raw_group in raw_groups:
            if not isinstance(raw_group, dict):
                continue
            edge = str(raw_group.get("edge", "")).lower()
            if edge not in boxes or not isinstance(raw_group.get("clicks"), list):
                continue
            left, top, right, bottom = boxes[edge]
            clicks = []
            for raw_click in raw_group["clicks"]:
                if not isinstance(raw_click, dict):
                    continue
                x, y, label = (
                    raw_click.get("x"),
                    raw_click.get("y"),
                    raw_click.get("label"),
                )
                if (
                    isinstance(x, bool)
                    or not isinstance(x, (int, float))
                    or isinstance(y, bool)
                    or not isinstance(y, (int, float))
                    or label not in (0, 1)
                    or not 0.0 <= float(x) <= 1.0
                    or not 0.0 <= float(y) <= 1.0
                ):
                    continue
                clicks.append(
                    {
                        "x": (left + float(x) * (right - left)) / width,
                        "y": (top + float(y) * (bottom - top)) / height,
                        "label": int(label),
                    }
                )
            if not any(click["label"] == 1 for click in clicks):
                continue
            groups.append(
                {
                    "id": len(groups) + 1,
                    "source": "E",
                    "description": str(raw_group.get("description", "")),
                    "clicks": clicks,
                }
            )
    return groups, {
        "n_proposed": len(groups),
        "reference_offset_frames": neighbour_offset,
        "edge_fraction": 0.30,
    }


def _find_mask_guided_groups(
    frame: np.ndarray,
    existing_masks: list[dict[str, Any]],
    *,
    raw_index: int,
    fps: float,
    temporal_offsets: list[int],
    frame_dir: Path,
    model: str,
    run_border_scan: bool = True,
    discovery_pass_index: int = 1,
    discovery_pass_count: int = 1,
    visual_qa_focus: str = "",
    prior_attempts: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Ask specifically for life not covered by persistent accepted masks.

    The general S3 finder remains a recall fallback, but this pass uses the
    repository's production SoM contract: the target has translucent green
    accepted masks and a coordinate grid, while nearby raw video frames supply
    temporal evidence.  This makes the MLLM spend its attention on genuinely
    missed instances instead of repeatedly rediscovering known animals.
    """
    guided_dir = frame_dir / "mask_guided"
    guided_dir.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    raw_target_path = guided_dir / "target_raw.png"
    cv2.imwrite(str(raw_target_path), frame)
    overlay = render_existing_masks_overlay(frame, existing_masks, alpha=0.18)
    overlay_path = guided_dir / "existing_masks.png"
    grid_path = guided_dir / "existing_masks_grid.png"
    strong_overlay_path = guided_dir / "existing_masks_strong.png"
    outline_overlay_path = guided_dir / "existing_masks_outline.png"
    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(grid_path), render_grid_overlay(overlay))
    strong_overlay = render_existing_masks_overlay(
        frame, existing_masks, alpha=0.48
    )
    cv2.imwrite(str(strong_overlay_path), strong_overlay)
    outline_overlay = frame.copy()
    for result in existing_masks:
        mask = np.asarray(result["mask"]).astype(bool)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            outline_overlay, contours, -1, (0, 255, 0), 3, cv2.LINE_AA
        )
    cv2.imwrite(str(outline_overlay_path), render_grid_overlay(outline_overlay))
    neighbours = P.extract_neighbours(
        raw_index, temporal_offsets[:2], str(guided_dir), fps=fps
    )
    (
        focus_left,
        focus_right,
        focus_top,
        focus_bottom,
        focus_region,
    ) = _discovery_focus_region(discovery_pass_index, discovery_pass_count)
    if discovery_pass_count <= 0:
        if discovery_pass_index <= 4:
            pass_instruction = (
                f"This is required convergence sweep pass "
                f"{discovery_pass_index} of 4. Exhaustively scan the "
                f"{focus_region} (normalized x={focus_left:.2f}.."
                f"{focus_right:.2f}, y={focus_top:.2f}..{focus_bottom:.2f}) "
                "and report every missed target within that tile. "
                "An empty tile does not end convergence; all four tiles will be checked "
                "before the model may stop. Still report an unmistakable "
                "missed target elsewhere if one is visible. "
            )
        else:
            pass_instruction = (
                f"This is full-frame convergence discovery pass "
                f"{discovery_pass_index}. Scan the entire residual frame "
                "carefully for visible life not already covered by green masks. "
                "If every detectable organism is already covered, return no "
                "click groups; that is the stop decision. Otherwise report "
                "every missed target you can see. "
            )
    if discovery_pass_count > 0:
        pass_instruction = (
            f"This is discovery pass {discovery_pass_index} of "
            f"{discovery_pass_count}. Scan the {focus_region} especially carefully "
            f"(normalized x={focus_left:.2f}..{focus_right:.2f}, "
            f"y={focus_top:.2f}..{focus_bottom:.2f}) so successive passes "
            "cover the full frame. Green masks include everything accepted by earlier "
            "passes. Still report an obvious missed target elsewhere. "
        )
    if visual_qa_focus.strip():
        pass_instruction += (
            "A human visual-QA reviewer identified this priority residual: "
            f"{visual_qa_focus.strip()} Inspect it especially carefully, while "
            "still rejecting substrate and still reporting other obvious missed "
            "life. You must choose and verify all click coordinates yourself. "
        )
    prior_attempts = list(prior_attempts or [])
    if prior_attempts:
        compact_attempts = []
        for attempt in prior_attempts[-24:]:
            description = str(attempt.get("description", "target")).strip()
            click = attempt.get("click") or {}
            compact_attempts.append(
                f"{description} near "
                f"({float(click.get('x', 0.0)):.3f},"
                f"{float(click.get('y', 0.0)):.3f})"
            )
        pass_instruction += (
            "These candidates were already evaluated in earlier convergence "
            "passes: " + "; ".join(compact_attempts) + ". Do not repeat the "
            "same identity at the same location. If one truly remains uncovered "
            "because the earlier coordinate was wrong, use a visibly different, "
            "verified interior click on the actual residual structure. "
        )
    has_focus_crops = not (
        focus_left == 0.0
        and focus_right == 1.0
        and focus_top == 0.0
        and focus_bottom == 1.0
    )
    focus_paths: list[Path] = []
    if has_focus_crops:
        left = int(round(focus_left * width))
        right = int(round(focus_right * width))
        top = int(round(focus_top * height))
        bottom = int(round(focus_bottom * height))
        focus_raw = cv2.resize(
            frame[top:bottom, left:right],
            (width, height),
            interpolation=cv2.INTER_CUBIC,
        )
        strong_grid = render_grid_overlay(strong_overlay)
        focus_strong = cv2.resize(
            strong_grid[top:bottom, left:right],
            (width, height),
            interpolation=cv2.INTER_CUBIC,
        )
        focus_raw_path = guided_dir / "focus_raw.png"
        focus_strong_path = guided_dir / "focus_strong_grid.png"
        cv2.imwrite(str(focus_raw_path), focus_raw)
        cv2.imwrite(str(focus_strong_path), focus_strong)
        focus_paths = [focus_raw_path, focus_strong_path]
    user_text = _mask_guided_discovery_prompt(
        pass_instruction=pass_instruction,
        has_focus_crops=has_focus_crops,
    )
    content = [
        {"type": "image", "image": str(raw_target_path)},
        {"type": "image", "image": str(grid_path)},
        {"type": "image", "image": str(strong_overlay_path)},
        {"type": "image", "image": str(outline_overlay_path)},
    ]
    content.extend(
        {"type": "image", "image": str(path)} for path in focus_paths
    )
    content.extend({"type": "image", "image": path} for _label, path in neighbours)
    content.append({"type": "text", "text": user_text})
    response = P.send_claude_request(
        [
            {
                "role": "system",
                "content": load_click_discovery_system_prompt("underwater"),
            },
            {"role": "user", "content": content},
        ],
        model=model,
        max_tokens=P.response_token_budget(
            model, 1600, sonnet5_minimum=8192
        ),
    )
    (guided_dir / "response.txt").write_text(
        response or "<none>", encoding="utf-8"
    )
    groups = parse_creature_click_groups(response or "")
    for index, group in enumerate(groups, 1):
        group["id"] = index
        group["source"] = "M"
    border_meta = {
        "n_proposed": 0,
        "reference_offset_frames": None,
        "edge_fraction": 0.30,
    }
    if run_border_scan:
        border_groups, border_meta = _find_border_groups(
            frame,
            overlay,
            raw_index=raw_index,
            temporal_offsets=temporal_offsets,
            guided_dir=guided_dir,
            model=model,
        )
        groups.extend(border_groups)
    for index, group in enumerate(groups, 1):
        group["id"] = index
    proposed_overlay = render_proposed_click_groups_overlay(
        frame, groups, existing_masks=existing_masks
    )
    cv2.imwrite(str(guided_dir / "proposed_click_groups.png"), proposed_overlay)
    _write_json(guided_dir / "proposed_click_groups.json", groups)
    click_counts = _click_counts(
        [click for group in groups for click in group.get("clicks") or []]
    )
    return groups, {
        "n_proposed": len(groups),
        "n_positive_clicks": click_counts["positive"],
        "n_negative_clicks": click_counts["negative"],
        "n_reference_frames": len(neighbours),
        "persistent_known_masks": len(existing_masks),
        "focus_region": focus_region,
        "focus_x_range": [focus_left, focus_right],
        "focus_y_range": [focus_top, focus_bottom],
        "border_scan_enabled": run_border_scan,
        "border_scan": border_meta,
        "visual_qa_focus": visual_qa_focus.strip() or None,
        "n_prior_attempts_in_prompt": len(prior_attempts[-24:]),
    }


def _parse_same_identity_groups(
    answer: dict[str, Any],
    identity_components: list[list[int]],
) -> list[list[int]]:
    """Validate disjoint 1-based same-identity subgroups from Claude.

    Each reported subgroup must be wholly contained in one geometry-triggered
    review component. A component can contain several unrelated organisms, so
    callers must never infer that the whole component is one identity.
    """
    raw_groups = answer.get("same_identity_groups")
    if not isinstance(raw_groups, list):
        return []
    valid_components = [set(component) for component in identity_components]
    accepted: list[list[int]] = []
    used: set[int] = set()
    for raw_group in raw_groups:
        if not isinstance(raw_group, list):
            continue
        try:
            group = sorted({int(item) for item in raw_group})
        except (TypeError, ValueError):
            continue
        group_set = set(group)
        if (
            len(group) < 2
            or group_set.intersection(used)
            or not any(group_set.issubset(component) for component in valid_components)
        ):
            continue
        accepted.append(group)
        used.update(group_set)
    return accepted


def _verify_text_masks_batch(
    results: list[dict[str, Any]],
    known_masks: list[dict[str, Any]],
    frame: np.ndarray,
    width: int,
    height: int,
    output_dir: Path,
    *,
    model: str,
    temporal_context: list[tuple[str, str]] | None = None,
    motion_context: list[tuple[str, np.ndarray]] | None = None,
    identity_components: list[list[int]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Verify proposals, or partition geometry-triggered identity components."""
    if not results:
        return [], [], {"batch_size": 0, "fallback_count": 0}
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry_lines: list[str] = []
    for left_id in range(1, len(results) + 1):
        left_mask = np.asarray(results[left_id - 1]["mask"]).astype(bool)
        for right_id in range(left_id + 1, len(results) + 1):
            right_mask = np.asarray(results[right_id - 1]["mask"]).astype(bool)
            intersection = int(np.logical_and(left_mask, right_mask).sum())
            union = int(np.logical_or(left_mask, right_mask).sum())
            smaller = min(int(left_mask.sum()), int(right_mask.sum()))
            iou = intersection / union if union else 0.0
            smaller_overlap = intersection / smaller if smaller else 0.0
            geometry_lines.append(
                f"MASK {left_id} vs MASK {right_id}: intersection="
                f"{intersection} px, IoU={iou:.6f}, smaller-mask-overlap="
                f"{smaller_overlap:.6f}."
            )
    for candidate_id, result in enumerate(results, 1):
        candidate = np.asarray(result["mask"]).astype(bool)
        best_iou = 0.0
        best_smaller_overlap = 0.0
        for known in known_masks:
            prior = np.asarray(known["mask"]).astype(bool)
            intersection = int(np.logical_and(candidate, prior).sum())
            union = int(np.logical_or(candidate, prior).sum())
            smaller = min(int(candidate.sum()), int(prior.sum()))
            best_iou = max(best_iou, intersection / union if union else 0.0)
            best_smaller_overlap = max(
                best_smaller_overlap,
                intersection / smaller if smaller else 0.0,
            )
        geometry_lines.append(
            f"MASK {candidate_id} vs any prior accepted mask: max-IoU="
            f"{best_iou:.6f}, max-smaller-mask-overlap="
            f"{best_smaller_overlap:.6f}."
        )
    # Dense optical flow is useful for ordinary candidate verification, but it
    # proved actively misleading for identity partitioning: stationary adjacent
    # organisms share camera motion, while one flexible/elongated colony can
    # have very different local flow. Relationship mode therefore uses the raw
    # shared temporal crops rather than lossy scalar motion summaries.
    if identity_components is None:
        motion_context = list(motion_context or [])
        base_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for motion_label, motion_frame in motion_context:
            if motion_frame.shape[:2] != frame.shape[:2]:
                continue
            neighbour_gray = cv2.cvtColor(motion_frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                base_gray, neighbour_gray, None,
                0.5, 5, 25, 5, 7, 1.5, 0,
            )
            medians: list[np.ndarray] = []
            dispersions: list[np.ndarray] = []
            for candidate_id, result in enumerate(results, 1):
                candidate = np.asarray(result["mask"]).astype(bool)
                values = flow[candidate]
                if not len(values):
                    median = np.zeros(2, dtype=float)
                    mad = np.zeros(2, dtype=float)
                else:
                    median = np.median(values, axis=0)
                    mad = np.median(np.abs(values - median), axis=0)
                medians.append(median)
                dispersions.append(mad)
                geometry_lines.append(
                    f"MASK {candidate_id} target-to-{motion_label} median optical flow="
                    f"({median[0]:.2f},{median[1]:.2f}) px, component MAD="
                    f"({mad[0]:.2f},{mad[1]:.2f}) px."
                )
            for left_id in range(1, len(medians) + 1):
                for right_id in range(left_id + 1, len(medians) + 1):
                    delta = float(np.linalg.norm(
                        medians[left_id - 1] - medians[right_id - 1]
                    ))
                    within = float(max(
                        0.25,
                        np.linalg.norm(dispersions[left_id - 1]),
                        np.linalg.norm(dispersions[right_id - 1]),
                    ))
                    geometry_lines.append(
                        f"MASK {left_id} vs MASK {right_id} at {motion_label}: "
                        f"median-flow separation={delta:.2f} px, largest within-mask "
                        f"flow MAD={within:.2f} px, separation/MAD={delta / within:.2f}."
                    )
    geometry_sentence = (
        "Deterministic candidate-mask geometry (exact pixels, not a visual "
        "estimate): " + " ".join(geometry_lines) + " "
        if geometry_lines else ""
    )
    context = frame.copy()
    for batch_id, result in enumerate(results, 1):
        mask = np.asarray(result["mask"]).astype(bool)
        # Skip the palette's conventional accepted-mask green for candidates.
        color = COLORS_BGR[batch_id % len(COLORS_BGR)]
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(context, contours, -1, color, 3, cv2.LINE_AA)
        ys, xs = np.nonzero(mask)
        if len(xs):
            label_x = max(8, min(width - 44, int(np.median(xs))))
            label_y = max(28, min(height - 8, int(np.median(ys))))
            cv2.putText(
                context, str(batch_id), (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3,
                cv2.LINE_AA,
            )
            cv2.putText(
                context, str(batch_id), (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 1, cv2.LINE_AA,
            )
    context_path = output_dir / "text_proposal_full_context.png"
    cv2.imwrite(str(context_path), context)
    cell_width, cell_height = 400, 300
    panels: list[np.ndarray] = []
    for batch_id, result in enumerate(results, 1):
        mask = np.asarray(result["mask"]).astype(bool)
        left, top, crop_width, crop_height = P._mask_crop_geom(
            mask, [], width, height, 0.22
        )
        crop = frame[top : top + crop_height, left : left + crop_width].copy()
        crop_mask = mask[top : top + crop_height, left : left + crop_width]
        scale = min(
            (cell_width - 12) / max(1, crop_width),
            (cell_height - 42) / max(1, crop_height),
        )
        resized_size = (
            max(1, int(round(crop_width * scale))),
            max(1, int(round(crop_height * scale))),
        )
        zoom = cv2.resize(crop, resized_size, interpolation=cv2.INTER_CUBIC)
        zoom_mask = cv2.resize(
            crop_mask.astype(np.uint8), resized_size,
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        contours, _ = cv2.findContours(
            zoom_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(zoom, contours, -1, (255, 255, 0), 2, cv2.LINE_AA)
        panel = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
        panel[36 : 36 + zoom.shape[0], 6 : 6 + zoom.shape[1]] = zoom
        label = f"MASK {batch_id}: {result.get('source_prompt', '')}"
        cv2.putText(
            panel, label[:52], (7, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
        panels.append(panel)

    columns = min(3, len(panels))
    rows = []
    for start in range(0, len(panels), columns):
        row = panels[start : start + columns]
        row.extend(
            np.zeros_like(panels[0]) for _ in range(columns - len(row))
        )
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    sheet_path = output_dir / "text_proposal_batch_verify.png"
    cv2.imwrite(str(sheet_path), sheet)
    temporal_context = list(temporal_context or [])
    temporal_labels = "; ".join(label for label, _path in temporal_context)
    temporal_frames = [
        (label, cv2.imread(path)) for label, path in temporal_context
    ]
    temporal_frames = [
        (label, image) for label, image in temporal_frames if image is not None
    ]
    temporal_identity_path: Path | None = None
    if temporal_frames:
        identity_cell_width, identity_cell_height = 360, 280
        identity_rows: list[np.ndarray] = []
        for batch_id, result in enumerate(results, 1):
            mask = np.asarray(result["mask"]).astype(bool)
            left, top, crop_width, crop_height = P._mask_crop_geom(
                mask, [], width, height, 0.18
            )
            sources = [("TARGET", frame), *temporal_frames]
            cells: list[np.ndarray] = []
            for source_index, (label, source_frame) in enumerate(sources):
                crop = source_frame[
                    top : top + crop_height,
                    left : left + crop_width,
                ].copy()
                if source_index == 0:
                    crop_mask = mask[
                        top : top + crop_height,
                        left : left + crop_width,
                    ]
                    contours, _ = cv2.findContours(
                        crop_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    cv2.drawContours(
                        crop, contours, -1, (255, 0, 255), 3, cv2.LINE_AA
                    )
                scale = min(
                    (identity_cell_width - 12) / max(1, crop.shape[1]),
                    (identity_cell_height - 42) / max(1, crop.shape[0]),
                )
                resized = cv2.resize(
                    crop,
                    (
                        max(1, int(round(crop.shape[1] * scale))),
                        max(1, int(round(crop.shape[0] * scale))),
                    ),
                    interpolation=cv2.INTER_CUBIC,
                )
                cell = np.zeros(
                    (identity_cell_height, identity_cell_width, 3),
                    dtype=np.uint8,
                )
                cell[36 : 36 + resized.shape[0], 6 : 6 + resized.shape[1]] = resized
                cv2.putText(
                    cell, f"MASK {batch_id} | {label}"[:54], (7, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1,
                    cv2.LINE_AA,
                )
                cells.append(cell)
            identity_rows.append(np.hstack(cells))
        temporal_identity_sheet = np.vstack(identity_rows)
        temporal_identity_path = output_dir / "text_proposal_temporal_identity.png"
        cv2.imwrite(str(temporal_identity_path), temporal_identity_sheet)
    relation_identity_path: Path | None = None
    relation_row_images: list[np.ndarray] = []
    if identity_components is not None and temporal_frames:
        relation_cell_width, relation_cell_height = 420, 320
        relation_rows: list[np.ndarray] = []
        for component_number, component in enumerate(identity_components, 1):
            component_masks = [
                np.asarray(results[batch_id - 1]["mask"]).astype(bool)
                for batch_id in component
            ]
            union_mask = np.logical_or.reduce(component_masks)
            left, top, crop_width, crop_height = P._mask_crop_geom(
                union_mask, [], width, height, 0.30
            )
            sources = [("TARGET", frame), *temporal_frames]
            cells: list[np.ndarray] = []
            for source_index, (label, source_frame) in enumerate(sources):
                crop = source_frame[
                    top : top + crop_height,
                    left : left + crop_width,
                ].copy()
                if source_index == 0:
                    for batch_id, mask in zip(component, component_masks):
                        crop_mask = mask[
                            top : top + crop_height,
                            left : left + crop_width,
                        ]
                        color = COLORS_BGR[batch_id % len(COLORS_BGR)]
                        contours, _ = cv2.findContours(
                            crop_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )
                        cv2.drawContours(
                            crop, contours, -1, color, 3, cv2.LINE_AA
                        )
                        ys, xs = np.nonzero(crop_mask)
                        if len(xs):
                            label_x = max(8, min(crop_width - 38, int(np.median(xs))))
                            label_y = max(28, min(crop_height - 8, int(np.median(ys))))
                            cv2.putText(
                                crop, str(batch_id), (label_x, label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (255, 255, 255), 3, cv2.LINE_AA,
                            )
                            cv2.putText(
                                crop, str(batch_id), (label_x, label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                color, 1, cv2.LINE_AA,
                            )
                scale = min(
                    (relation_cell_width - 12) / max(1, crop.shape[1]),
                    (relation_cell_height - 42) / max(1, crop.shape[0]),
                )
                resized = cv2.resize(
                    crop,
                    (
                        max(1, int(round(crop.shape[1] * scale))),
                        max(1, int(round(crop.shape[0] * scale))),
                    ),
                    interpolation=cv2.INTER_CUBIC,
                )
                cell = np.zeros(
                    (relation_cell_height, relation_cell_width, 3),
                    dtype=np.uint8,
                )
                cell[36 : 36 + resized.shape[0], 6 : 6 + resized.shape[1]] = resized
                heading = (
                    f"COMPONENT {component_number} {component} | {label}"
                )
                cv2.putText(
                    cell, heading[:62], (7, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (255, 255, 255), 1, cv2.LINE_AA,
                )
                cells.append(cell)
            relation_row = np.hstack(cells)
            relation_rows.append(relation_row)
            relation_row_images.append(relation_row)
        relation_identity_sheet = np.vstack(relation_rows)
        relation_identity_path = output_dir / "identity_component_temporal.png"
        cv2.imwrite(str(relation_identity_path), relation_identity_sheet)
    temporal_sentence = (
        "The THIRD image is a temporal identity sheet: one row per candidate "
        "with the identical crop at TARGET, earlier, and later times; magenta "
        "outlines the candidate only in the TARGET column. Trace real structure "
        "and occlusion changes across each row. "
        f"The next {len(temporal_context)} labeled images are unmarked full video "
        f"frames ({temporal_labels}) for parallax, occlusion, and depth "
        "reasoning. "
        if temporal_identity_path is not None else ""
    )
    if identity_components is not None:
        component_text = "; ".join(
            "[" + ", ".join(str(item) for item in component) + "]"
            for component in identity_components
        )
        prompt = (
            "The FIRST image is the raw full frame for identity and depth context. "
            "The SECOND image is a full-frame contour map on untouched raw pixels: "
            "each already-accepted biological mask has a distinct colored outline "
            "and numeric ID. "
            "The THIRD image is the decisive component relationship sheet. Each "
            "row uses ONE SHARED CROP for every member of an ambiguous component "
            "at TARGET, earlier, and later times. Only TARGET has numbered colored "
            "outlines. Trace branch/body continuity, gaps, occlusions, and depth "
            "inside a row; do not compare differently cropped panels as if they "
            "were spatially aligned. The FOURTH image is a per-mask temporal sheet "
            "for inspecting each candidate's extent. The following labeled images "
            "are unmarked full video frames for wider depth context. "
            + "The LAST image contains numbered zoomed panels with each mask "
            "outlined in cyan. Every listed mask has ALREADY passed strict review "
            "as real marine life; do not keep, reject, or relabel masks here. Your "
            "only task is to identify masks that are fragments or duplicate views "
            "of the SAME INDIVIDUAL biological identity and therefore should be "
            "combined. Geometry only triggered these ambiguous review components: "
            f"{component_text}. A component can contain several unrelated organisms. "
            "Report only the smallest same-identity SUBGROUPS within a component; "
            "never combine the entire component merely because its members form a "
            "contact chain. Omitted IDs remain separate. Multiple disjoint subgroups "
            "inside one component are allowed. Group masks only when you can trace "
            "actual continuous branch/body structure at the same depth, or when they "
            "substantially duplicate the same visible pixels and identity. Do not "
            "group masks merely because they touch, overlap in 2D, have the same "
            "taxon/texture/color, or lie close together. Preserve distinct foreground "
            "and background organisms across occlusions. Use temporal parallax, "
            "occlusion ordering, apparent independent motion, and parallax in the "
            "actual temporal images. Co-motion alone never proves one identity: "
            "neighboring stationary life and substrate move together with the "
            "camera. Apparent local motion differences also do not override a "
            "directly visible continuous flexible or elongated biological structure. "
            "At every proposed connection, compare the structure on both sides. An "
            "abrupt transition between a densely branched fan and a smooth whip, or "
            "a discontinuity in color, thickness, orientation, texture, or depth, is "
            "evidence for separate touching/occluding organisms. Pixel overlap at a "
            "base or crossing is not anatomical continuity. One identity must have a "
            "plausibly continuous central axis or branching system through the join. "
            "Conversely, small or zero pixel overlap is NOT evidence against one "
            "identity: independently generated segmentation fragments often meet "
            "edge-to-edge. Trace the centerline and tangent through the boundary. A "
            "near-collinear continuation with compatible thickness, texture, and "
            "branch morphology on both sides is strong continuity evidence even "
            "when only a few pixels touch. Do not call such aligned fragments "
            "crossing stalks unless two distinct axes visibly continue past the "
            "intersection. "
            "For every proposed group, first trace an actual continuous connection "
            "in the THIRD image's TARGET cell, then confirm that interpretation in "
            "its earlier/later cells. If there is no visible connection, omit it. "
            "When uncertain, omit the group so the originals are safely preserved. "
            + geometry_sentence
            + "\n\nOutput brief reasoning followed by EXACTLY ONE trailing tag:\n"
            '<answer>{"same_identity_groups":[[1,3],[4,5]]}</answer>\n'
            "Use an empty list when no masks should be combined."
        )
    else:
        prompt = (
        "The FIRST image is the raw full frame for identity and depth context. "
        "The SECOND image is a full-frame contour map on untouched raw pixels: "
        "each candidate has a distinct colored outline and numeric ID. Prior "
        "accepted masks are deliberately NOT drawn because exact overlap against "
        "them has already been measured and reported below. No region is "
        "color-filled, so judge depth from original texture, brightness, and "
        "occlusion. "
        + temporal_sentence
        + "The LAST image contains numbered zoomed underwater panels, each with "
        + "one candidate SAM3 mask outlined in cyan. For EVERY MASK ID, decide "
        "whether cyan covers "
        "genuinely NEW real, distinct marine life: an animal, coherent animal "
        "colony, sponge, coral, macroalga, seagrass, or other living organism. "
        "Underwater life is often camouflaged. Set keep=false if cyan is part "
        "of, an appendage of, or a duplicate view of the same organism already "
        "accepted earlier. Also reject bare substrate, debris, shadow, or other "
        "non-living background. Keep cyan when it is distinct life near or "
        "attached to other life. A kept mask must cover one COMPLETE biological "
        "identity: reject a branch, appendage, or patch when its same colony "
        "visibly continues outside cyan. It must also cover exactly ONE "
        "identity/depth layer: reject unions across occlusions or distinct "
        "branching systems, even when both are the same taxon. Inspect all "
        "non-frame-edge mask boundaries for cut-off continuous branches. Use "
        "the full-frame candidate map to distinguish adjacent colonies. Do NOT "
        "call two masks duplicates merely because their zoomed crops look "
        "similar or the organisms share a taxon; duplicates must substantially "
        "cover the same pixels AND the same biological identity. A claim that "
        "masks overlap or duplicate one another must agree with the exact pixel "
        "geometry below. Near-disjoint masks may touch at a boundary but are "
        "not mask duplicates. Return one entry per ID. "
        "Taxonomic, color, or texture similarity does NOT imply one biological "
        "identity. Candidates at different depth layers remain separate when "
        "they merely touch or overlap in 2D projection; use temporal parallax, "
        "occlusion ordering, independent motion, and branch/body continuity "
        "rather than fill-color contact to decide. Do not invent continuation "
        "behind a real occluder: a mask can be complete for the organism's "
        "VISIBLE extent when its remaining extent is outside the frame or truly "
        "occluded. Before claiming that adjacent candidates are one continuous "
        "identity, trace an actual branch/body connection across their boundary "
        "without crossing an occlusion or depth discontinuity; nearby similar "
        "branches alone are not continuity. "
        "Use the measured optical-flow evidence below: a consistent pairwise "
        "median-flow separation that is large relative to within-mask flow MAD "
        "is positive evidence for different depths or independently moving "
        "identities. Do not call such candidates one stationary continuous "
        "organism merely because they have similar morphology. "
        + geometry_sentence
        + "\n\n"
        + "Output brief reasoning followed by EXACTLY ONE trailing tag:\n"
        '<answer>{"masks":[{"id":1,"keep":true,'
        '"confidence":0.0,"complete_identity":true,'
        '"single_identity":true}]}</answer>'
        )
    (output_dir / "text_proposal_batch_verify_prompt.txt").write_text(
        prompt, encoding="utf-8"
    )
    response = P.send_claude_request(
        [{"role": "user", "content": [
            {"type": "image", "image": str(output_dir.parent / "target.png")},
            {"type": "image", "image": str(context_path)},
            *(
                [{"type": "image", "image": str(relation_identity_path)}]
                if relation_identity_path is not None else []
            ),
            *(
                [{"type": "image", "image": str(temporal_identity_path)}]
                if temporal_identity_path is not None else []
            ),
            *[
                {"type": "image", "image": path}
                for _label, path in temporal_context
            ],
            {"type": "image", "image": str(sheet_path)},
            {"type": "text", "text": prompt},
        ]}],
        model=model,
        max_tokens=P.response_token_budget(model, 1200),
    )
    (output_dir / "text_proposal_batch_verify.txt").write_text(
        response or "<none>", encoding="utf-8"
    )
    answer = P._extract_answer_json(response) or {}
    if identity_components is not None:
        proposed_groups = _parse_same_identity_groups(answer, identity_components)
        groups: list[list[int]] = []
        confirmations: list[dict[str, Any]] = []
        for confirmation_index, group in enumerate(proposed_groups, 1):
            component_index = next(
                (
                    index for index, component in enumerate(identity_components)
                    if set(group).issubset(component)
                ),
                None,
            )
            confirmation: dict[str, Any] = {
                "group": group,
                "component_index": component_index,
                "accepted": False,
            }
            if (
                component_index is None
                or component_index >= len(relation_row_images)
            ):
                confirmation["terminal_reason"] = "missing_temporal_relation_row"
                confirmations.append(confirmation)
                continue
            confirm_path = (
                output_dir / f"identity_group_{confirmation_index}_confirm.png"
            )
            cv2.imwrite(
                str(confirm_path), relation_row_images[component_index]
            )
            ids_text = ", ".join(str(item) for item in group)
            confirm_prompt = (
                "The FIRST image is the untouched full frame. The SECOND image is "
                "one shared crop at TARGET, earlier, and later times. TARGET has "
                "numbered colored outlines; the other times are raw. Decide only "
                f"whether masks [{ids_text}] are fragments/duplicates of exactly "
                "ONE biological individual or colony and should be unioned. Other "
                "numbered masks, if present, are context and must remain separate. "
                "A true union requires an actually visible anatomical connection, "
                "a plausibly continuous central axis or branching system through "
                "the join, compatible morphology on both sides, and one depth "
                "layer. Mere pixel contact/overlap, co-motion, proximity, or a "
                "shared base location is insufficient. An abrupt fan-to-whip, "
                "thick-to-thin, color, texture, orientation, or depth transition "
                "means separate touching/occluding organisms. Small or zero overlap "
                "does not disqualify edge-to-edge fragments when one compatible "
                "centerline continues nearly collinearly through the boundary. When "
                "uncertain, "
                "answer false so the original masks are preserved.\n\n"
                "Output brief reasoning followed by EXACTLY ONE trailing tag:\n"
                '<answer>{"same_identity":true,"confidence":0.0,'
                '"visible_connection":true,"same_branching_system":true,'
                '"single_depth_layer":true}</answer>'
            )
            (output_dir / f"identity_group_{confirmation_index}_confirm_prompt.txt").write_text(
                confirm_prompt, encoding="utf-8"
            )
            confirm_response = P.send_claude_request(
                [{"role": "user", "content": [
                    {
                        "type": "image",
                        "image": str(output_dir.parent / "target.png"),
                    },
                    {"type": "image", "image": str(confirm_path)},
                    {"type": "text", "text": confirm_prompt},
                ]}],
                model=model,
                max_tokens=P.response_token_budget(model, 600),
            )
            (output_dir / f"identity_group_{confirmation_index}_confirm.txt").write_text(
                confirm_response or "<none>", encoding="utf-8"
            )
            confirm_answer = P._extract_answer_json(confirm_response) or {}
            try:
                confidence = min(
                    1.0, max(0.0, float(confirm_answer.get("confidence")))
                )
            except (TypeError, ValueError):
                confidence = 0.0
            accepted = (
                confirm_answer.get("same_identity") is True
                and confirm_answer.get("visible_connection") is True
                and confirm_answer.get("same_branching_system") is True
                and confirm_answer.get("single_depth_layer") is True
                and confidence >= 0.75
            )
            confirmation.update({
                "accepted": accepted,
                "confidence": confidence,
                "verdict": confirm_answer,
                "terminal_reason": (
                    "focused_relation_confirmed"
                    if accepted else "focused_relation_rejected"
                ),
            })
            confirmations.append(confirmation)
            if accepted:
                groups.append(group)
        raw_groups = answer.get("same_identity_groups")
        return list(results), [], {
            "batch_size": len(results),
            "relation_mode": True,
            "relation_answer_valid": isinstance(raw_groups, list),
            "proposed_same_identity_groups": proposed_groups,
            "same_identity_groups": groups,
            "focused_confirmations": confirmations,
            "fallback_count": 0,
        }
    decisions: dict[int, dict[str, Any]] = {}
    raw_decisions = answer.get("masks")
    if isinstance(raw_decisions, list):
        for item in raw_decisions:
            if not isinstance(item, dict) or item.get("keep") not in (True, False):
                continue
            try:
                batch_id = int(item.get("id"))
                confidence = min(1.0, max(0.0, float(item.get("confidence"))))
            except (TypeError, ValueError):
                continue
            if 1 <= batch_id <= len(results):
                decisions[batch_id] = {
                    "keep": (
                        item["keep"] is True
                        and item.get("complete_identity") is True
                        and item.get("single_identity") is True
                    ),
                    "confidence": confidence,
                    "complete_identity": item.get("complete_identity") is True,
                    "single_identity": item.get("single_identity") is True,
                }

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for batch_id, result in enumerate(results, 1):
        decision = decisions.get(batch_id)
        if decision is None:
            missing.append(result)
            continue
        result["creature_confidence"] = float(decision["confidence"])
        result["mask_complete_identity"] = bool(
            decision["complete_identity"]
        )
        result["mask_single_identity"] = bool(decision["single_identity"])
        result["text_verify_method"] = "batch"
        (kept if decision["keep"] else dropped).append(result)

    if missing:
        fallback_kept, fallback_dropped = verify_masks(
            missing,
            frame.copy(),
            width,
            height,
            str(output_dir),
            model=model,
            tag_prefix="vt_fallback",
            allow_all_life=True,
            strict_identity=True,
        )
        for result in fallback_kept + fallback_dropped:
            result["text_verify_method"] = "individual_fallback"
        kept.extend(fallback_kept)
        dropped.extend(fallback_dropped)
    return kept, dropped, {
        "batch_size": len(results),
        "batch_decision_count": len(decisions),
        "fallback_count": len(missing),
    }


def _consolidate_ambiguous_overlaps(
    known_masks: list[dict[str, Any]],
    *,
    frame: np.ndarray,
    width: int,
    height: int,
    frame_dir: Path,
    model: str,
    raw_index: int,
    temporal_offsets: list[int],
    service: Any,
    max_clicks: int,
    max_attempts: int,
    zoom_crop_frac: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Consolidate only temporally verified same-identity overlap subgroups.

    Geometry merely triggers review and may connect several distinct organisms
    into one component. Sonnet must explicitly partition only fragment/duplicate
    subgroups, after which each subgroup union must pass the ordinary strict
    complete/single-identity verifier. Any inconclusive gate preserves originals.
    """
    masks = [np.asarray(item["mask"]).astype(bool) for item in known_masks]
    components, pairs = _ambiguous_overlap_components(masks)
    metadata: dict[str, Any] = {
        "trigger_smaller_mask_overlap": 0.40,
        "trigger_max_gap_fraction": 0.006,
        "candidate_pairs": pairs,
        "components": components,
        "candidate_mask_indices": [],
        "batch_kept_indices": [],
        "batch_dropped_indices": [],
        "same_identity_groups": [],
        "union_attempts": [],
        "n_consolidated_components": 0,
        "n_masks_removed": 0,
    }
    audit_dir = frame_dir / "overlap_identity_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    if not components:
        _write_json(audit_dir / "audit_summary.json", metadata)
        return known_masks, metadata

    candidate_indices = sorted({index for group in components for index in group})
    metadata["candidate_mask_indices"] = candidate_indices
    candidate_results: list[dict[str, Any]] = []
    for index in candidate_indices:
        candidate = dict(known_masks[index])
        candidate["source_prompt"] = f"accepted mask {index + 1}"
        candidate["overlap_audit_original_index"] = index
        candidate_results.append(candidate)
    candidate_set = set(candidate_indices)
    original_to_batch_id = {
        original_index: batch_index
        for batch_index, original_index in enumerate(candidate_indices, 1)
    }
    identity_components = [
        [original_to_batch_id[index] for index in component]
        for component in components
    ]
    noncandidate_masks = [
        item for index, item in enumerate(known_masks) if index not in candidate_set
    ]

    temporal_dir = audit_dir / "temporal_context"
    temporal_dir.mkdir(parents=True, exist_ok=True)
    temporal_context = P.extract_neighbours(
        raw_index,
        temporal_offsets[:1],
        str(temporal_dir),
        fps=P.SRC.get("fps", 30.0),
    )
    motion_context: list[tuple[str, np.ndarray]] = []
    if temporal_offsets:
        nearest = int(temporal_offsets[0])
        fps = float(P.SRC.get("fps", 30.0))
        for signed_offset in (-nearest, nearest):
            neighbour_index = raw_index + signed_offset
            if neighbour_index < 0:
                continue
            try:
                neighbour = P.read_video_frame(neighbour_index)
            except SystemExit:
                continue
            motion_context.append((f"{signed_offset / fps:+.1f}s", neighbour))
    batch_kept, batch_dropped, batch_meta = _verify_text_masks_batch(
        candidate_results,
        noncandidate_masks,
        frame.copy(),
        width,
        height,
        audit_dir,
        model=model,
        temporal_context=temporal_context,
        motion_context=motion_context,
        identity_components=identity_components,
    )
    kept_indices = {
        int(item["overlap_audit_original_index"]) for item in batch_kept
    }
    dropped_indices = {
        int(item["overlap_audit_original_index"]) for item in batch_dropped
    }
    metadata["batch_verifier"] = batch_meta
    metadata["batch_kept_indices"] = sorted(kept_indices)
    metadata["batch_dropped_indices"] = sorted(dropped_indices)

    relation_groups: list[list[int]] = []
    for batch_group in batch_meta.get("same_identity_groups", []):
        relation_groups.append([
            candidate_indices[int(batch_id) - 1] for batch_id in batch_group
        ])
    metadata["same_identity_groups"] = relation_groups

    replacements: dict[int, dict[str, Any]] = {}
    removed: set[int] = set()
    grouped_indices = {index for group in relation_groups for index in group}
    for component in components:
        if not set(component).intersection(grouped_indices):
            metadata["union_attempts"].append({
                "component": component,
                "relation_group": [],
                "union_verified": False,
                "terminal_reason": "relation_classifier_kept_separate",
            })

    for component_number, component in enumerate(relation_groups, 1):
        review_component = next(
            review for review in components if set(component).issubset(review)
        )
        audit_record: dict[str, Any] = {
            "component": review_component,
            "relation_group": component,
            "union_verified": False,
        }
        union_mask = np.logical_or.reduce([masks[index] for index in component])
        union_result = {
            "creature_id": component_number,
            "description": "candidate union of temporally reviewed overlapping masks",
            "mask": union_mask,
            "clicks_used": [],
            "source": "overlap_identity_union",
        }
        union_kept, union_dropped = verify_masks(
            [union_result],
            frame.copy(),
            width,
            height,
            str(audit_dir),
            model=model,
            tag_prefix=f"union_component_{component_number}",
            allow_all_life=True,
            strict_identity=True,
        )
        if union_kept:
            anchor = min(component)
            consolidated = dict(union_kept[0])
            consolidated["source"] = "overlap_identity_union"
            consolidated["consolidated_original_indices"] = list(component)
            replacements[anchor] = consolidated
            removed.update(index for index in component if index != anchor)
            audit_record["union_verified"] = True
            audit_record["terminal_reason"] = "strict_union_accepted"
        else:
            union_failure = (
                union_dropped[0].get("mask_quality_failure")
                if union_dropped else "missing_verdict"
            )
            audit_record["union_failure"] = union_failure
            repair_click = (
                union_dropped[0].get("mask_quality_repair_click")
                if union_dropped else None
            )
            repair_allowed = False
            if isinstance(repair_click, dict) and repair_click.get("label") == 1:
                repair_x = min(
                    width - 1,
                    max(0, int(round(float(repair_click["x"]) * (width - 1)))),
                )
                repair_y = min(
                    height - 1,
                    max(0, int(round(float(repair_click["y"]) * (height - 1)))),
                )
                outside_distance = cv2.distanceTransform(
                    np.logical_not(union_mask).astype(np.uint8),
                    cv2.DIST_L2,
                    3,
                )[repair_y, repair_x]
                normalized_distance = float(outside_distance) / float(
                    np.hypot(width, height)
                )
                repair_allowed = normalized_distance <= 0.10
                audit_record["repair_click"] = dict(repair_click)
                audit_record["repair_click_distance_from_union"] = (
                    normalized_distance
                )
            if union_failure == "fragment" and repair_allowed:
                repair_clicks: list[dict[str, Any]] = []
                for index in component:
                    anchor_x, anchor_y = _mask_anchor(masks[index])
                    candidate = {
                        "x": anchor_x / max(1, width - 1),
                        "y": anchor_y / max(1, height - 1),
                        "label": 1,
                    }
                    if not P._duplicate_click(repair_clicks, candidate, tolerance=0.01):
                        repair_clicks.append(candidate)
                if not P._duplicate_click(repair_clicks, repair_click, tolerance=0.01):
                    repair_clicks.append(dict(repair_click))
                repair_group = {
                    "id": component_number,
                    "description": (
                        "complete visible extent of one temporally verified "
                        "overlapping marine-life identity"
                    ),
                    "source": "overlap_identity_repair",
                    "clicks": repair_clicks,
                }
                repair_dir = (
                    audit_dir / f"union_component_{component_number}_repair"
                )
                repair_dir.mkdir(parents=True, exist_ok=True)
                repaired, repair_trace = P.refine_group_mm_hybrid(
                    service,
                    repair_group,
                    str(frame_dir / "target.png"),
                    frame.copy(),
                    width,
                    height,
                    str(repair_dir),
                    max_clicks=max_clicks,
                    max_attempts=max_attempts,
                    strict_quality=True,
                    zoom_crop_frac=zoom_crop_frac,
                )
                audit_record["repair_attempted"] = True
                audit_record["repair_trace"] = repair_trace
                repaired_kept: list[dict[str, Any]] = []
                repaired_dropped: list[dict[str, Any]] = []
                if np.asarray(repaired["mask"]).astype(bool).any():
                    repaired_kept, repaired_dropped = verify_masks(
                        [repaired],
                        frame.copy(),
                        width,
                        height,
                        str(audit_dir),
                        model=model,
                        tag_prefix=f"repaired_component_{component_number}",
                        allow_all_life=True,
                        strict_identity=True,
                    )
                if repaired_kept:
                    anchor = min(component)
                    consolidated = dict(repaired_kept[0])
                    consolidated["source"] = "overlap_identity_repair"
                    consolidated["consolidated_original_indices"] = list(component)
                    replacements[anchor] = consolidated
                    removed.update(
                        index for index in component if index != anchor
                    )
                    audit_record["repair_verified"] = True
                    audit_record["terminal_reason"] = "strict_repair_accepted"
                else:
                    audit_record["repair_verified"] = False
                    audit_record["repair_failure"] = (
                        repaired_dropped[0].get("mask_quality_failure")
                        if repaired_dropped else "empty_or_missing_verdict"
                    )
                    audit_record["terminal_reason"] = "strict_repair_rejected"
            else:
                audit_record["repair_attempted"] = False
                audit_record["terminal_reason"] = "strict_union_rejected"
        metadata["union_attempts"].append(audit_record)

    consolidated_masks = [
        replacements.get(index, item)
        for index, item in enumerate(known_masks)
        if index not in removed
    ]
    metadata["n_consolidated_components"] = len(replacements)
    metadata["n_masks_removed"] = len(removed)
    _write_json(audit_dir / "audit_summary.json", metadata)
    return consolidated_masks, metadata


def _recover_group_batch(
    groups: list[dict[str, Any]],
    *,
    service: Any,
    target_path: Path,
    frame: np.ndarray,
    width: int,
    height: int,
    stage_dir: Path,
    model: str,
    known_masks: list[dict[str, Any]],
    max_clicks: int,
    max_attempts: int,
    min_recovery_confidence: float,
    mask_generator: str,
    zoom_crop_frac: float,
) -> dict[str, Any]:
    """Refine and verify one discovery pass against the current mask context."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    pristine_frame = frame.copy()
    groups, dedup_removed = B._geom_dedup(groups, width, height)
    for index, group in enumerate(groups, 1):
        group["id"] = index

    generated: list[dict[str, Any]] = []
    generator = {
        "hybrid": P.refine_group_mm_hybrid,
        "full": P.refine_group_mm,
        "zoom": P.refine_group_mm_zoom,
    }[mask_generator]
    for group in groups:
        generator_kwargs: dict[str, Any] = {
            "max_clicks": max_clicks,
            "max_attempts": max_attempts,
            "strict_quality": True,
        }
        if mask_generator == "zoom":
            generator_kwargs["seg_crop_frac"] = zoom_crop_frac
        elif mask_generator == "hybrid":
            generator_kwargs["zoom_crop_frac"] = zoom_crop_frac
        result, trace = generator(
            service,
            group,
            str(target_path),
            pristine_frame.copy(),
            width,
            height,
            str(stage_dir),
            **generator_kwargs,
        )
        result["source"] = group.get("source", "B")
        result["discovery_clicks"] = [
            dict(click) for click in group.get("original_clicks") or group["clicks"]
        ]
        result["initial_clicks"] = [dict(click) for click in group["clicks"]]
        result["seed_click"] = _first_positive_click(group)
        result["trace"] = trace
        generated.append(result)

    verified_before_confidence, dropped = verify_masks(
        generated,
        pristine_frame.copy(),
        width,
        height,
        str(stage_dir),
        model=model,
        allow_all_life=True,
        strict_identity=True,
    )
    # A strict verifier sometimes diagnoses a real target more accurately than
    # the inner candidate picker (for example, an incomplete branch or a mask
    # merged into a neighbour). Feed that diagnosis back into SAM3 instead of
    # throwing away the target. The verifier controls when this loop ends by
    # either accepting the mask or omitting an actionable repair click. Exact
    # duplicate clicks are the only deterministic no-progress stop.
    postverify_repair_attempts = 0
    postverify_repaired = 0
    repair_pending = list(dropped)
    terminal_dropped: list[dict[str, Any]] = []
    repair_round = 1
    while repair_pending:
        repair_dir = stage_dir / f"postverify_repair_round_{repair_round}"
        repair_dir.mkdir(parents=True, exist_ok=True)
        repair_generated: list[dict[str, Any]] = []
        for rejected in repair_pending:
            repair_click = rejected.get("mask_quality_repair_click")
            prior_clicks = [
                dict(click)
                for click in rejected.get("clicks_used") or []
                if isinstance(click, dict) and click.get("label") in (0, 1)
            ]
            if (
                not isinstance(repair_click, dict)
                # A click within 2.5% of an earlier same-label click is not a
                # materially new correction. This is a no-progress guard, not a
                # ceiling on how many useful clicks the agent may request.
                or P._duplicate_click(prior_clicks, repair_click, tolerance=0.025)
            ):
                terminal_dropped.append(rejected)
                continue
            repair_group = {
                "id": int(rejected.get("creature_id", 0)),
                "description": rejected.get("description", ""),
                "source": rejected.get("source", "B"),
                "clicks": [*prior_clicks, dict(repair_click)],
            }
            generator_kwargs: dict[str, Any] = {
                "max_clicks": max_clicks,
                "max_attempts": max_attempts,
                "strict_quality": True,
            }
            if mask_generator == "zoom":
                generator_kwargs["seg_crop_frac"] = zoom_crop_frac
            elif mask_generator == "hybrid":
                generator_kwargs["zoom_crop_frac"] = zoom_crop_frac
            repaired, repair_trace = generator(
                service,
                repair_group,
                str(target_path),
                pristine_frame.copy(),
                width,
                height,
                str(repair_dir),
                **generator_kwargs,
            )
            postverify_repair_attempts += 1
            repaired["source"] = rejected.get("source", "B")
            repaired["discovery_clicks"] = [
                dict(click)
                for click in rejected.get("discovery_clicks")
                or rejected.get("initial_clicks")
                or prior_clicks
            ]
            repaired["initial_clicks"] = [
                dict(click)
                for click in rejected.get("initial_clicks") or prior_clicks
            ]
            repaired["seed_click"] = rejected.get("seed_click")
            repaired["trace"] = [
                *list(rejected.get("trace") or []),
                {
                    "postverify_repair_round": repair_round,
                    "failure": rejected.get("mask_quality_failure"),
                    "repair_click": dict(repair_click),
                },
                *repair_trace,
            ]
            repaired["postverify_repair_history"] = [
                *list(rejected.get("postverify_repair_history") or []),
                {
                    "round": repair_round,
                    "failure": rejected.get("mask_quality_failure"),
                    "click": dict(repair_click),
                },
            ]
            repaired["postverify_repair_round"] = repair_round
            if np.asarray(repaired["mask"]).astype(bool).any():
                repair_generated.append(repaired)
            else:
                terminal_dropped.append(repaired)
        if not repair_generated:
            break
        repair_kept, repair_rejected = verify_masks(
            repair_generated,
            pristine_frame.copy(),
            width,
            height,
            str(repair_dir),
            model=model,
            allow_all_life=True,
            strict_identity=True,
        )
        postverify_repaired += len(repair_kept)
        verified_before_confidence.extend(repair_kept)
        repair_pending = repair_rejected
        repair_round += 1
    dropped = terminal_dropped
    verified_before_confidence, mask_nms_removed = _mask_level_nms(
        verified_before_confidence
    )
    verified_before_confidence = [
        result
        for result in verified_before_confidence
        if np.asarray(result["mask"]).astype(bool).any()
    ]
    preliminary_matches = _match_firstpass(
        verified_before_confidence, known_masks
    )
    low_confidence_recoveries = [
        result
        for index, result in enumerate(verified_before_confidence)
        if index not in preliminary_matches
        and float(result.get("creature_confidence", 0.0))
        < min_recovery_confidence
    ]
    low_confidence_ids = {id(result) for result in low_confidence_recoveries}
    verified = [
        result
        for result in verified_before_confidence
        if id(result) not in low_confidence_ids
    ]
    matches = _match_firstpass(verified, known_masks)
    replacements_by_result_index = {
        index: replacement
        for index, match in matches.items()
        if (
            replacement := _replacement_for_contained_mask(
                verified[index], match, known_masks
            )
        ) is not None
    }
    recovered = [
        result for index, result in enumerate(verified) if index not in matches
    ]

    _draw_diagnostic(
        pristine_frame,
        known_masks,
        verified,
        matches,
        dropped + low_confidence_recoveries,
        stage_dir / "diagnostic.png",
    )

    click_items: list[dict[str, Any]] = []
    for index, result in enumerate(verified):
        click = result.get("seed_click") or {}
        initial_clicks = [
            dict(item) for item in result.get("initial_clicks") or []
        ]
        discovery_clicks = [
            dict(item) for item in result.get("discovery_clicks") or initial_clicks
        ]
        final_clicks = [
            dict(item) for item in result.get("clicks_used") or []
            if isinstance(item, dict) and item.get("label") in (0, 1)
        ]
        initial_counts = _click_counts(initial_clicks)
        final_counts = _click_counts(final_clicks)
        match = matches.get(index)
        matched_known = (
            known_masks[int(match["firstpass_index"])] if match else None
        )
        click_items.append(
            {
                "status": (
                    "replaced"
                    if index in replacements_by_result_index
                    else "matched" if match else "recovered"
                ),
                "source": result.get("source", "B"),
                "description": result.get("description", ""),
                "creature_confidence": float(
                    result.get("creature_confidence", 0.0)
                ),
                "area_px": int(np.asarray(result["mask"]).sum()),
                "click": {
                    "x": float(click.get("x", 0.0)),
                    "y": float(click.get("y", 0.0)),
                },
                "initial_clicks": initial_clicks,
                "discovery_clicks": discovery_clicks,
                "final_clicks": final_clicks,
                "n_initial_positive_clicks": initial_counts["positive"],
                "n_initial_negative_clicks": initial_counts["negative"],
                "n_final_positive_clicks": final_counts["positive"],
                "n_final_negative_clicks": final_counts["negative"],
                "matched_firstpass_index": (
                    int(match["firstpass_index"])
                    if match and matched_known.get("source") == "firstpass"
                    else None
                ),
                "matched_known_index": (
                    int(match["firstpass_index"]) if match else None
                ),
                "matched_known_source": (
                    str(matched_known.get("source", "firstpass"))
                    if matched_known else None
                ),
                "iou_firstpass": (
                    float(match["iou"])
                    if match and matched_known.get("source") == "firstpass"
                    else 0.0
                ),
                "iou_known": float(match["iou"]) if match else 0.0,
                "match_method": str(match["method"]) if match else None,
                "overlap_coefficient": (
                    float(match.get("overlap_coefficient", 0.0))
                    if match else 0.0
                ),
                "replacement_area_ratio": (
                    float(replacements_by_result_index[index]["area_ratio"])
                    if index in replacements_by_result_index else None
                ),
            }
        )

    return {
        "n_groups": len(groups),
        "dedup_removed": dedup_removed,
        "verified_before_confidence": verified_before_confidence,
        "verified": verified,
        "dropped": dropped,
        "low_confidence": low_confidence_recoveries,
        "postverify_repair_attempts": postverify_repair_attempts,
        "postverify_repaired": postverify_repaired,
        "mask_nms_removed": mask_nms_removed,
        "matches": matches,
        "replacements": list(replacements_by_result_index.values()),
        "recovered": recovered,
        "click_items": click_items,
    }


def process_frame(
    record: dict[str, Any],
    *,
    repo_root: Path,
    firstpass_root: Path,
    initial_mask_root: Path | None,
    skip_firstpass: bool,
    output_root: Path,
    model: str,
    phrase_model: str,
    firstpass_model: str,
    taxon_labels: list[str],
    text_proposal_mode: str,
    text_proposal_threshold: float,
    min_text_confidence: float,
    finder_mode: str,
    strategy: str,
    temporal_offsets: list[int],
    temporal_seconds: list[float],
    max_clicks: int,
    max_attempts: int,
    mask_generator: str,
    max_click_groups_per_pass: int,
    zoom_crop_frac: float,
    click_localization_crop_frac: float,
    visual_qa_focus: str,
    mask_guided_passes: int,
    sparse_extra_pass: bool,
    border_scan: str,
    min_recovery_confidence: float,
    service: Any,
) -> dict[str, Any]:
    frame_id = str(record["id"])
    raw_index = int(record["frame_index"])
    frame_dir = output_root / frame_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    video_path = (repo_root / str(record["video"])).resolve()
    if skip_firstpass:
        firstpass: list[dict[str, Any]] = []
        firstpass_summary: dict[str, Any] = {
            "model": firstpass_model,
            "skipped": True,
        }
    else:
        firstpass, firstpass_summary = _load_firstpass(
            firstpass_root / frame_id, firstpass_model
        )
    frame, fps = _read_raw_frame(video_path, raw_index)
    effective_temporal_offsets = temporal_offsets
    if temporal_seconds:
        effective_temporal_offsets = sorted(
            {max(1, int(round(fps * seconds))) for seconds in temporal_seconds}
        )
    height, width = frame.shape[:2]
    initial_masks = _load_initial_masks(
        initial_mask_root, frame_id, (height, width)
    )
    if firstpass and firstpass[0]["mask"].shape != (height, width):
        raise RuntimeError(
            f"mask/frame size mismatch for {frame_id}: "
            f"{firstpass[0]['mask'].shape} vs {(height, width)}"
        )
    target_path = frame_dir / "target.png"
    if not cv2.imwrite(str(target_path), frame):
        raise RuntimeError(f"could not write target frame: {target_path}")

    P.SRC.update(video=str(video_path), frames_dir=None, frame_outputs=None, fps=fps)
    P.MODEL = model
    cfg = B.STRATEGIES[strategy]
    started = time.time()

    adaptive_prompt_meta: dict[str, Any] | None = None
    if text_proposal_mode == "adaptive":
        text_specs, adaptive_prompt_meta = _plan_adaptive_text_prompts(
            record,
            target_path,
            frame_dir / "text_proposals",
            model=phrase_model,
        )
    elif text_proposal_mode != "none":
        text_specs = production_prompt_bank(
            taxon_labels, mode=text_proposal_mode
        )
    else:
        text_specs = []
    text_sam3_started = time.time()
    text_candidates, text_meta = _propose_text_masks(
        service,
        target_path,
        text_specs,
        [*firstpass, *initial_masks],
        threshold=text_proposal_threshold,
        output_dir=frame_dir / "text_proposals",
    )
    text_sam3_runtime = time.time() - text_sam3_started
    text_verified_before_confidence: list[dict[str, Any]] = []
    text_rejected: list[dict[str, Any]] = []
    text_low_confidence: list[dict[str, Any]] = []
    text_verify_meta = {"batch_size": 0, "fallback_count": 0}
    text_verify_started = time.time()
    if text_candidates:
        text_temporal_dir = frame_dir / "text_proposals" / "temporal_context"
        text_temporal_dir.mkdir(parents=True, exist_ok=True)
        text_temporal_context = P.extract_neighbours(
            raw_index,
            effective_temporal_offsets[:1],
            str(text_temporal_dir),
            fps=fps,
        )
        text_motion_context: list[tuple[str, np.ndarray]] = []
        if effective_temporal_offsets:
            nearest_offset = int(effective_temporal_offsets[0])
            for signed_offset in (-nearest_offset, nearest_offset):
                neighbour_index = raw_index + signed_offset
                if neighbour_index < 0:
                    continue
                try:
                    neighbour_frame = P.read_video_frame(neighbour_index)
                except SystemExit:
                    continue
                text_motion_context.append(
                    (f"{signed_offset / fps:+.1f}s", neighbour_frame)
                )
        (
            text_verified_before_confidence,
            text_rejected,
            text_verify_meta,
        ) = _verify_text_masks_batch(
            text_candidates,
            [*firstpass, *initial_masks],
            frame.copy(),
            width,
            height,
            frame_dir / "text_proposals",
            model=model,
            temporal_context=text_temporal_context,
            motion_context=text_motion_context,
        )
        text_low_confidence = [
            result
            for result in text_verified_before_confidence
            if float(result.get("creature_confidence", 0.0))
            < min_text_confidence
        ]
    text_low_ids = {id(result) for result in text_low_confidence}
    text_verify_runtime = time.time() - text_verify_started
    text_verified = [
        result
        for result in text_verified_before_confidence
        if id(result) not in text_low_ids
        and np.asarray(result["mask"]).astype(bool).any()
    ]
    known_masks = [*firstpass, *initial_masks, *text_verified]
    text_meta.update(
        {
            "mode": text_proposal_mode,
            "sam3_runtime_sec": text_sam3_runtime,
            "mllm_verify_runtime_sec": text_verify_runtime,
            "verifier": "batch_with_individual_fallback",
            "verifier_metadata": text_verify_meta,
            "min_mllm_confidence": min_text_confidence,
            "n_verified_before_confidence": len(
                text_verified_before_confidence
            ),
            "n_verified": len(text_verified),
            "n_rejected": len(text_rejected),
            "n_low_confidence": len(text_low_confidence),
            "prompt_bank": [
                {
                    "text": spec.text,
                    "group": spec.group,
                    "source_taxon": spec.source_taxon,
                }
                for spec in text_specs
            ],
            "adaptive_prompt_plan": adaptive_prompt_meta,
        }
    )
    _write_json(frame_dir / "text_proposals" / "proposal_summary.json", text_meta)
    _draw_source_diagnostic(
        frame,
        [*firstpass, *initial_masks],
        text_verified,
        [],
        frame_dir / "text_proposals_verified.png",
        click_model=model,
    )

    initial_known_count = len(known_masks)
    effective_mask_guided_passes = choose_mask_guided_pass_count(
        mask_guided_passes,
        initial_known_count=initial_known_count,
        sparse_extra_pass=sparse_extra_pass,
        finder_mode=finder_mode,
    )
    all_verified_before_confidence: list[dict[str, Any]] = []
    all_verified: list[dict[str, Any]] = []
    all_dropped: list[dict[str, Any]] = []
    all_low_confidence: list[dict[str, Any]] = []
    recovered: list[dict[str, Any]] = []
    click_items: list[dict[str, Any]] = []
    total_groups = 0
    total_dedup_removed = 0
    total_mask_nms_removed = 0
    total_refound = 0
    total_replaced = 0
    click_gate_dropped = 0
    engine_meta = {
        "n_whole_frame": 0,
        "n_tile": 0,
        "tile_raw_candidates": 0,
        "tile_gate_dropped": 0,
        "dedup_removed": 0,
        "review_removed": 0,
        "review_moved": 0,
    }
    if finder_mode in {"hybrid", "s3"}:
        _base, _tile, cold_groups, engine_meta = B._engine_on_frame(
            frame,
            raw_index,
            width,
            height,
            cfg,
            str(frame_dir),
            fps,
            review=False,
            offsets=effective_temporal_offsets,
        )
        cold_groups, click_gate_dropped = verify_clicks(
            cold_groups,
            frame,
            width,
            height,
            str(frame_dir),
            allow_all_life=True,
        )
        cold_batch = _recover_group_batch(
            cold_groups,
            service=service,
            target_path=target_path,
            frame=frame,
            width=width,
            height=height,
            stage_dir=frame_dir / "cold_click_recovery",
            model=model,
            known_masks=known_masks,
            max_clicks=max_clicks,
            max_attempts=max_attempts,
            min_recovery_confidence=min_recovery_confidence,
            mask_generator=mask_generator,
            zoom_crop_frac=zoom_crop_frac,
        )
        total_groups += int(cold_batch["n_groups"])
        total_dedup_removed += int(cold_batch["dedup_removed"])
        total_mask_nms_removed += int(cold_batch["mask_nms_removed"])
        total_replaced += len(cold_batch["replacements"])
        total_refound += (
            len(cold_batch["matches"]) - len(cold_batch["replacements"])
        )
        all_verified_before_confidence.extend(
            cold_batch["verified_before_confidence"]
        )
        all_verified.extend(cold_batch["verified"])
        all_dropped.extend(cold_batch["dropped"])
        all_low_confidence.extend(cold_batch["low_confidence"])
        for item in cold_batch["click_items"]:
            click_items.append({"discovery_pass": "cold", **item})
        for replacement in cold_batch["replacements"]:
            known_masks[int(replacement["known_index"])] = replacement["result"]
        recovered.extend(cold_batch["recovered"])
        known_masks.extend(cold_batch["recovered"])
    mask_guided_meta = {
        "n_proposed": 0,
        "n_reference_frames": 0,
        "persistent_firstpass_masks": len(firstpass),
        "persistent_text_masks": len(text_verified),
    }
    if finder_mode in {"mask-guided", "hybrid"}:
        pass_meta = []
        total_proposed = 0
        convergence_mode = effective_mask_guided_passes == 0
        convergence_terminal_reason = "requested_pass_count_completed"
        prior_mask_guided_attempts: list[dict[str, Any]] = []
        pass_index = 1
        while convergence_mode or pass_index <= effective_mask_guided_passes:
            pass_started = time.time()
            mask_guided_groups, one_pass_meta = _find_mask_guided_groups(
                frame,
                known_masks,
                raw_index=raw_index,
                fps=fps,
                temporal_offsets=effective_temporal_offsets,
                frame_dir=frame_dir / f"mask_guided_pass_{pass_index}",
                model=model,
                run_border_scan=_should_run_border_scan(
                    border_scan,
                    convergence_mode=convergence_mode,
                    pass_index=pass_index,
                    requested_passes=effective_mask_guided_passes,
                ),
                discovery_pass_index=pass_index,
                discovery_pass_count=(
                    0 if convergence_mode else effective_mask_guided_passes
                ),
                visual_qa_focus=visual_qa_focus,
                prior_attempts=prior_mask_guided_attempts,
            )
            agent_proposed_before_repeat_filter = len(mask_guided_groups)
            total_proposed += agent_proposed_before_repeat_filter
            mask_guided_groups, repeated_prior_groups = (
                _filter_prior_attempt_groups(
                    mask_guided_groups, prior_mask_guided_attempts
                )
            )
            proposed_before_limit = len(mask_guided_groups)
            for proposed_group in mask_guided_groups:
                positive = _first_positive_click(proposed_group)
                prior_mask_guided_attempts.append(
                    {
                        "description": proposed_group.get(
                            "description", "target"
                        ),
                        "click": {
                            "x": float(positive.get("x", 0.0)),
                            "y": float(positive.get("y", 0.0)),
                        },
                    }
                )
            recovery_stage_dir = frame_dir / f"click_recovery_pass_{pass_index}"
            recovery_stage_dir.mkdir(parents=True, exist_ok=True)
            if max_click_groups_per_pass > 0:
                mask_guided_groups = mask_guided_groups[
                    :max_click_groups_per_pass
                ]
            one_pass_meta["diagnostic_group_cap"] = (
                max_click_groups_per_pass or None
            )
            one_pass_meta["n_agent_proposed_before_repeat_filter"] = (
                agent_proposed_before_repeat_filter
            )
            one_pass_meta["n_prior_attempt_repeats_dropped"] = len(
                repeated_prior_groups
            )
            one_pass_meta["n_omitted_by_diagnostic_cap"] = (
                proposed_before_limit - len(mask_guided_groups)
            )
            one_pass_meta["n_selected_for_localization"] = len(
                mask_guided_groups
            )
            localization_frame = render_existing_masks_overlay(
                frame, known_masks
            )
            mask_guided_groups, localization_dropped = verify_clicks(
                mask_guided_groups,
                localization_frame,
                width,
                height,
                str(recovery_stage_dir),
                strict=True,
                tag_prefix="localize",
                allow_all_life=True,
                correct_click=True,
                existing_mask_overlay=True,
                allow_existing_mask_expansion=bool(visual_qa_focus.strip()),
                region_frac=click_localization_crop_frac,
            )
            mask_guided_groups, conflicting_negatives_dropped = (
                _drop_conflicting_negative_clicks(
                    mask_guided_groups, width, height
                )
            )
            click_gate_dropped += localization_dropped
            one_pass_meta["n_click_localization_dropped"] = localization_dropped
            one_pass_meta["n_conflicting_negative_clicks_dropped"] = (
                conflicting_negatives_dropped
            )
            one_pass_meta["n_after_click_localization"] = len(mask_guided_groups)
            localized_clicks = [
                click
                for group in mask_guided_groups
                for click in group.get("clicks") or []
            ]
            localized_counts = _click_counts(localized_clicks)
            one_pass_meta["n_localized_positive_clicks"] = localized_counts[
                "positive"
            ]
            one_pass_meta["n_localized_negative_clicks"] = localized_counts[
                "negative"
            ]
            cv2.imwrite(
                str(recovery_stage_dir / "localized_click_groups.png"),
                render_proposed_click_groups_overlay(
                    frame, mask_guided_groups, existing_masks=known_masks
                ),
            )
            _write_json(
                recovery_stage_dir / "localized_click_groups.json",
                mask_guided_groups,
            )
            one_pass_meta["n_selected_for_recovery"] = len(
                mask_guided_groups
            )
            batch = _recover_group_batch(
                mask_guided_groups,
                service=service,
                target_path=target_path,
                frame=frame,
                width=width,
                height=height,
                stage_dir=recovery_stage_dir,
                model=model,
                known_masks=known_masks,
                max_clicks=max_clicks,
                max_attempts=max_attempts,
                min_recovery_confidence=min_recovery_confidence,
                mask_generator=mask_generator,
                zoom_crop_frac=zoom_crop_frac,
            )
            total_groups += int(batch["n_groups"])
            total_dedup_removed += int(batch["dedup_removed"])
            total_mask_nms_removed += int(batch["mask_nms_removed"])
            total_replaced += len(batch["replacements"])
            total_refound += len(batch["matches"]) - len(batch["replacements"])
            all_verified_before_confidence.extend(
                batch["verified_before_confidence"]
            )
            all_verified.extend(batch["verified"])
            all_dropped.extend(batch["dropped"])
            all_low_confidence.extend(batch["low_confidence"])
            for item in batch["click_items"]:
                click_items.append(
                    {"discovery_pass": pass_index, **item}
                )
            recovered.extend(batch["recovered"])
            for replacement in batch["replacements"]:
                known_masks[int(replacement["known_index"])] = replacement["result"]
            # This is the central iterative behavior: the next discovery pass sees
            # masks recovered by this pass as persistent green context.
            known_masks.extend(batch["recovered"])
            _write_json(
                frame_dir / "checkpoint_masks_rle.json",
                {
                    "frame_size_hw": [height, width],
                    "completed_mask_guided_passes": pass_index,
                    "masks": [
                        _encode_mask(np.asarray(item["mask"]).astype(bool))
                        for item in known_masks
                    ],
                },
            )
            pass_meta.append(
                {
                    "pass_index": pass_index,
                    **one_pass_meta,
                    "n_groups_after_dedup": int(batch["n_groups"]),
                    "n_dedup_removed": int(batch["dedup_removed"]),
                    "n_verified": len(batch["verified"]),
                    "n_refound": len(batch["matches"]) - len(batch["replacements"]),
                    "n_replaced": len(batch["replacements"]),
                    "n_recovered": len(batch["recovered"]),
                    "n_dropped": len(batch["dropped"])
                    + len(batch["low_confidence"]),
                    "runtime_sec": time.time() - pass_started,
                }
            )
            if (
                convergence_mode
                and pass_index > 4
                and proposed_before_limit == 0
            ):
                convergence_terminal_reason = (
                    "agent_reported_no_missed_life"
                    if agent_proposed_before_repeat_filter == 0
                    else "agent_reported_only_previously_evaluated_life"
                )
                break
            pass_index += 1
        mask_guided_meta = {
            "n_passes": len(pass_meta),
            "requested_passes": mask_guided_passes,
            "convergence_mode": convergence_mode,
            "terminal_reason": (
                convergence_terminal_reason
            ),
            "sparse_extra_pass_applied": (
                not convergence_mode
                and
                effective_mask_guided_passes > mask_guided_passes
            ),
            "n_proposed_before_dedup": total_proposed,
            "n_reference_frames": sum(
                int(item.get("n_reference_frames", 0)) for item in pass_meta
            ),
            "persistent_firstpass_masks": len(firstpass),
            "persistent_text_masks": len(text_verified),
            "passes": pass_meta,
        }
    known_masks, overlap_identity_audit = _consolidate_ambiguous_overlaps(
        known_masks,
        frame=frame,
        width=width,
        height=height,
        frame_dir=frame_dir,
        model=model,
        raw_index=raw_index,
        temporal_offsets=effective_temporal_offsets,
        service=service,
        max_clicks=max_clicks,
        max_attempts=max_attempts,
        zoom_crop_frac=zoom_crop_frac,
    )
    _write_json(
        frame_dir / "checkpoint_masks_rle.json",
        {
            "frame_size_hw": [height, width],
            "completed_mask_guided_passes": int(
                mask_guided_meta.get("n_passes", 0)
            ),
            "overlap_identity_audit_complete": True,
            "masks": [
                _encode_mask(np.asarray(item["mask"]).astype(bool))
                for item in known_masks
            ],
        },
    )
    final_masks = [np.asarray(item["mask"]).astype(bool) for item in known_masks]
    render_frame = cv2.imread(str(target_path))
    if render_frame is None:
        raise RuntimeError(f"could not reload target frame: {target_path}")
    _draw_numbered_masks(
        render_frame, final_masks, frame_dir / "presentation_overlay.png"
    )
    _draw_source_diagnostic(
        render_frame,
        [*firstpass, *initial_masks],
        text_verified,
        recovered,
        frame_dir / "source_diagnostic.png",
        click_model=model,
    )
    _draw_source_diagnostic(
        render_frame,
        [*firstpass, *initial_masks],
        text_verified,
        recovered,
        frame_dir / "diagnostic.png",
        click_model=model,
    )

    text_items = [
        {
            "source": "T",
            "prompt": str(result.get("source_prompt", "")),
            "prompt_group": str(result.get("source_group", "")),
            "source_taxon": str(result.get("source_taxon", "")),
            "verify_method": str(result.get("text_verify_method", "")),
            "sam3_score": float(result.get("sam3_score", 0.0)),
            "creature_confidence": float(
                result.get("creature_confidence", 0.0)
            ),
            "area_px": int(np.asarray(result["mask"]).sum()),
        }
        for result in text_verified
    ]

    result = {
        "frame_id": frame_id,
        "video": str(record["video"]),
        "raw_frame_index": raw_index,
        "time_seconds": float(record.get("time_seconds", raw_index / fps)),
        "model": model,
        "phrase_model": phrase_model,
        "anthropic_effort": os.environ.get("ANTHROPIC_EFFORT") or None,
        "firstpass_model": str(firstpass_summary.get("model", "")),
        "firstpass_skipped": skip_firstpass,
        "initial_mask_root": (
            str(initial_mask_root) if initial_mask_root is not None else None
        ),
        "text_proposal_mode": text_proposal_mode,
        "text_proposal_threshold": text_proposal_threshold,
        "min_text_confidence": min_text_confidence,
        "finder_mode": finder_mode,
        "mask_guided_passes": int(mask_guided_meta.get("n_passes", 0)),
        "requested_mask_guided_passes": mask_guided_passes,
        "sparse_extra_pass": sparse_extra_pass,
        "sparse_extra_pass_applied": (
            effective_mask_guided_passes > 0
            and
            effective_mask_guided_passes > mask_guided_passes
        ),
        "border_scan": border_scan,
        "persistent_click_masks_between_passes": True,
        "strategy": strategy,
        "whole_frame_review": False,
        "mask_generator": mask_generator,
        "max_click_groups_per_pass": max_click_groups_per_pass,
        "zoom_crop_frac": zoom_crop_frac,
        "click_localization_crop_frac": click_localization_crop_frac,
        "visual_qa_focus": visual_qa_focus or None,
        "temporal_offsets": effective_temporal_offsets,
        "temporal_seconds": temporal_seconds,
        "n_firstpass": len(firstpass),
        "n_initial_masks": len(initial_masks),
        "n_text_candidates": len(text_candidates),
        "n_text_verified": len(text_verified),
        "n_text_dropped": len(text_rejected) + len(text_low_confidence),
        "n_known_before_click": initial_known_count,
        "n_click_groups_after_gate": total_groups,
        "n_click_gate_dropped": click_gate_dropped,
        "n_combined_click_dedup_removed": total_dedup_removed,
        "mask_guided": mask_guided_meta,
        "min_recovery_confidence": min_recovery_confidence,
        "n_click_masks_verified_before_recovery_confidence": len(
            all_verified_before_confidence
        ),
        "n_click_masks_verified": len(all_verified),
        "n_click_masks_dropped": len(all_dropped) + len(all_low_confidence),
        "n_recovery_confidence_dropped": len(all_low_confidence),
        "n_mask_nms_removed": total_mask_nms_removed,
        "overlap_identity_audit": overlap_identity_audit,
        "n_refound": total_refound,
        "n_replaced": total_replaced,
        "n_recovered": len(recovered),
        "n_final": len(final_masks),
        "engine": engine_meta,
        "text_proposals": text_meta,
        "text_items": text_items,
        "click_items": click_items,
        "runtime_sec": time.time() - started,
        "api_failed": _api_failed(frame_dir),
        "presentation_overlay": "presentation_overlay.png",
        "diagnostic": "diagnostic.png",
        "source_diagnostic": "source_diagnostic.png",
    }
    _write_json(frame_dir / "result.json", result)
    _write_json(frame_dir / "final_masks_rle.json", {
        "frame_size_hw": [height, width],
        "masks": [_encode_mask(mask) for mask in final_masks],
    })
    return result


def main() -> None:
    args = parse_args()
    phrase_model = args.phrase_model or args.model
    if args.effort:
        os.environ["ANTHROPIC_EFFORT"] = args.effort
    repo_root = Path.cwd().resolve()
    manifest_path = (repo_root / args.manifest).resolve()
    manifest = _read_json(manifest_path)
    selection_value = args.selection_manifest or str(
        manifest.get("selection_manifest", "")
    )
    taxa_by_clip: dict[str, list[str]] = {}
    selection_path: Path | None = None
    if args.text_proposal_mode != "none":
        if not selection_value:
            raise SystemExit(
                "text proposals require --selection-manifest or the benchmark "
                "manifest's selection_manifest field"
            )
        selection_path = Path(selection_value)
        if not selection_path.is_absolute():
            selection_path = (repo_root / selection_path).resolve()
        taxa_by_clip = build_taxa_by_clip(_read_json(selection_path))
    requested = set(args.frame_id)
    records = [record for record in manifest["frames"]
               if not requested or str(record["id"]) in requested]
    if requested - {str(record["id"]) for record in records}:
        missing = sorted(requested - {str(record["id"]) for record in records})
        raise SystemExit(f"frame ids not found in manifest: {missing}")

    firstpass_root = (repo_root / args.firstpass_root).resolve()
    initial_mask_root: Path | None = None
    if args.initial_mask_root:
        initial_mask_root = Path(args.initial_mask_root)
        if not initial_mask_root.is_absolute():
            initial_mask_root = (repo_root / initial_mask_root).resolve()
    output_root = (repo_root / args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    temporal_offsets = [int(value) for value in args.temporal_offsets.split(",") if value]
    temporal_seconds = [
        float(value) for value in args.temporal_seconds.split(",") if value
    ]
    if not temporal_offsets and not temporal_seconds:
        raise SystemExit("--temporal-offsets must contain at least one integer")
    if any(value <= 0 for value in temporal_seconds):
        raise SystemExit("--temporal-seconds values must be positive")
    if args.mask_guided_passes < 0:
        raise SystemExit("--mask-guided-passes must be non-negative")
    if args.max_clicks < 0:
        raise SystemExit("--max-clicks must be non-negative")
    if args.max_click_groups_per_pass < 0:
        raise SystemExit("--max-click-groups-per-pass must be non-negative")
    if not 0.1 <= args.zoom_crop_frac <= 1.0:
        raise SystemExit("--zoom-crop-frac must be in [0.1, 1.0]")
    if not 0.1 <= args.click_localization_crop_frac <= 1.0:
        raise SystemExit(
            "--click-localization-crop-frac must be in [0.1, 1.0]"
        )
    if not 0.0 <= args.min_recovery_confidence <= 1.0:
        raise SystemExit("--min-recovery-confidence must be in [0, 1]")
    if not 0.0 <= args.text_proposal_threshold <= 1.0:
        raise SystemExit("--text-proposal-threshold must be in [0, 1]")
    if not 0.0 <= args.min_text_confidence <= 1.0:
        raise SystemExit("--min-text-confidence must be in [0, 1]")

    print(f"Building SAM3 click-mode service for {len(records)} frame(s)...", flush=True)
    service = P.build_sam3_service()
    rows = []
    for record in records:
        frame_id = str(record["id"])
        result_path = output_root / frame_id / "result.json"
        if args.resume and result_path.exists():
            cached = _read_json(result_path)
            if (
                not cached.get("api_failed")
                and cached.get("model") == args.model
                and cached.get("phrase_model") == phrase_model
                and cached.get("anthropic_effort") == (args.effort or None)
                and cached.get("firstpass_model") == args.firstpass_model
                and bool(cached.get("firstpass_skipped")) == args.skip_firstpass
                and cached.get("initial_mask_root") == (
                    str(initial_mask_root)
                    if initial_mask_root is not None else None
                )
                and cached.get("text_proposal_mode") == args.text_proposal_mode
                and float(cached.get("text_proposal_threshold", -1.0))
                == args.text_proposal_threshold
                and float(cached.get("min_text_confidence", -1.0))
                == args.min_text_confidence
                and cached.get("finder_mode") == args.finder_mode
                and cached.get("persistent_click_masks_between_passes") is True
                and int(cached.get("requested_mask_guided_passes", -1))
                == args.mask_guided_passes
                and bool(cached.get("sparse_extra_pass"))
                == args.sparse_extra_pass
                and cached.get("border_scan") == args.border_scan
                and cached.get("mask_generator") == args.mask_generator
                and int(cached.get("max_click_groups_per_pass", -1))
                == args.max_click_groups_per_pass
                and float(cached.get("zoom_crop_frac", -1.0))
                == args.zoom_crop_frac
                and float(cached.get("click_localization_crop_frac", -1.0))
                == args.click_localization_crop_frac
                and cached.get("visual_qa_focus") == (
                    args.visual_qa_focus or None
                )
                and float(cached.get("min_recovery_confidence", -1.0))
                == args.min_recovery_confidence
            ):
                print(f"[{frame_id}] cached", flush=True)
                rows.append(cached)
                continue
        print(
            f"[{frame_id}] running "
            f"{'seeded masks only' if args.skip_firstpass else args.firstpass_model + ' first-pass'} + "
            f"SAM3 {args.text_proposal_mode} text ({phrase_model}) + "
            f"{args.model} click flow",
            flush=True,
        )
        result = process_frame(
            record,
            repo_root=repo_root,
            firstpass_root=firstpass_root,
            initial_mask_root=initial_mask_root,
            skip_firstpass=args.skip_firstpass,
            output_root=output_root,
            model=args.model,
            phrase_model=phrase_model,
            firstpass_model=args.firstpass_model,
            taxon_labels=taxa_by_clip.get(frame_id, []),
            text_proposal_mode=args.text_proposal_mode,
            text_proposal_threshold=args.text_proposal_threshold,
            min_text_confidence=args.min_text_confidence,
            finder_mode=args.finder_mode,
            strategy=args.strategy,
            temporal_offsets=temporal_offsets,
            temporal_seconds=temporal_seconds,
            max_clicks=args.max_clicks,
            max_attempts=args.max_attempts,
            mask_generator=args.mask_generator,
            max_click_groups_per_pass=args.max_click_groups_per_pass,
            zoom_crop_frac=args.zoom_crop_frac,
            click_localization_crop_frac=args.click_localization_crop_frac,
            visual_qa_focus=args.visual_qa_focus,
            mask_guided_passes=args.mask_guided_passes,
            sparse_extra_pass=args.sparse_extra_pass,
            border_scan=args.border_scan,
            min_recovery_confidence=args.min_recovery_confidence,
            service=service,
        )
        rows.append(result)
        print(
            f"[{frame_id}] first={result['n_firstpass']} "
            f"text={result['n_text_verified']} "
            f"refound={result['n_refound']} replaced={result['n_replaced']} "
            f"recovered={result['n_recovered']} "
            f"final={result['n_final']} dropped={result['n_click_masks_dropped']}",
            flush=True,
        )

    summary = {
        "benchmark_id": manifest.get("benchmark_id"),
        "model": args.model,
        "phrase_model": phrase_model,
        "anthropic_effort": args.effort or None,
        "firstpass_model": args.firstpass_model,
        "firstpass_root": str(firstpass_root),
        "firstpass_skipped": args.skip_firstpass,
        "initial_mask_root": (
            str(initial_mask_root) if initial_mask_root is not None else None
        ),
        "selection_manifest": str(selection_path) if selection_path else None,
        "text_proposal_mode": args.text_proposal_mode,
        "text_proposal_threshold": args.text_proposal_threshold,
        "min_text_confidence": args.min_text_confidence,
        "finder_mode": args.finder_mode,
        "strategy": args.strategy,
        "whole_frame_review": False,
        "mask_generator": args.mask_generator,
        "max_click_groups_per_pass": args.max_click_groups_per_pass,
        "zoom_crop_frac": args.zoom_crop_frac,
        "click_localization_crop_frac": args.click_localization_crop_frac,
        "visual_qa_focus": args.visual_qa_focus or None,
        "persistent_firstpass_mask_guidance": True,
        "persistent_click_masks_between_passes": True,
        "mask_guided_passes": args.mask_guided_passes,
        "sparse_extra_pass": args.sparse_extra_pass,
        "border_scan": args.border_scan,
        "min_recovery_confidence": args.min_recovery_confidence,
        "temporal_offsets": temporal_offsets,
        "temporal_seconds": temporal_seconds,
        "frames": rows,
        "api_failure_count": sum(bool(row.get("api_failed")) for row in rows),
        "exploratory_not_scored": True,
        "source_provenance": {
            "git": git_state(repo_root),
            "sha256": {
                str(Path(__file__).resolve().relative_to(repo_root)): sha256(
                    Path(__file__).resolve()
                ),
                "scripts/click_engine_probe.py": sha256(
                    repo_root / "scripts/click_engine_probe.py"
                ),
                "scripts/probe_sam3_life_prompts.py": sha256(
                    repo_root / "scripts/probe_sam3_life_prompts.py"
                ),
                "sam3/agent/client_claude.py": sha256(
                    repo_root / "sam3/agent/client_claude.py"
                ),
            },
        },
    }
    _write_json(output_root / "summary.json", summary)
    print(f"Wrote {output_root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
