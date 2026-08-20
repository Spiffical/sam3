#!/usr/bin/env python3
"""Summarize repeated custom-flow runs without treating mask count as recall."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


METRICS = (
    "n_firstpass",
    "n_text_verified",
    "n_recovered",
    "n_final",
    "runtime_sec",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Directory containing repeat_*/summary.json")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) >= 2 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def summarize(root: Path) -> dict[str, Any]:
    paths = sorted(root.glob("repeat_*/summary.json"))
    if len(paths) < 3:
        raise ValueError(f"expected at least 3 repeats under {root}; found {len(paths)}")
    docs = [read_json(path) for path in paths]
    frame_orders = [[str(row["frame_id"]) for row in doc["frames"]] for doc in docs]
    if any(order != frame_orders[0] for order in frame_orders[1:]):
        raise ValueError("repeat summaries do not contain the same ordered frame IDs")

    config_keys = (
        "model",
        "firstpass_model",
        "text_proposal_mode",
        "text_proposal_threshold",
        "min_text_confidence",
        "finder_mode",
        "mask_guided_passes",
        "sparse_extra_pass",
        "border_scan",
        "min_recovery_confidence",
        "persistent_click_masks_between_passes",
    )
    configs = [{key: doc.get(key) for key in config_keys} for doc in docs]
    if any(config != configs[0] for config in configs[1:]):
        raise ValueError("repeat configuration mismatch")

    frames: list[dict[str, Any]] = []
    for frame_index, frame_id in enumerate(frame_orders[0]):
        rows = [doc["frames"][frame_index] for doc in docs]
        metrics = {
            key: stats([float(row[key]) for row in rows]) for key in METRICS
        }
        metrics["text_sam3_runtime_sec"] = stats(
            [float(row["text_proposals"].get("sam3_runtime_sec", 0.0)) for row in rows]
        )
        metrics["text_verify_runtime_sec"] = stats(
            [
                float(row["text_proposals"].get("mllm_verify_runtime_sec", 0.0))
                for row in rows
            ]
        )
        frames.append(
            {
                "frame_id": frame_id,
                "metrics": metrics,
                "repeat_values": [
                    {key: row[key] for key in METRICS} for row in rows
                ],
            }
        )

    return {
        "schema_version": 1,
        "exploratory_not_scored": True,
        "warning": (
            "Mask counts are a coverage proxy, not recall. Use rendered-mask "
            "adjudication or ground truth for recall/precision claims."
        ),
        "root": str(root),
        "repeat_count": len(docs),
        "repeat_summaries": [str(path) for path in paths],
        "configuration": configs[0],
        "api_failure_count": sum(int(doc.get("api_failure_count", 0)) for doc in docs),
        "frames": frames,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Custom-flow repeat summary",
        "",
        f"Repeats: {summary['repeat_count']}. Exploratory, not scored.",
        "Mask count is a coverage proxy, not recall.",
        "",
        "| frame | first pass | text | click recovery | final | runtime s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["frames"]:
        metrics = row["metrics"]
        values = []
        for key in METRICS:
            item = metrics[key]
            values.append(f"{item['mean']:.2f} ± {item['std']:.2f}")
        lines.append(
            f"| {row['frame_id']} | {values[0]} | {values[1]} | {values[2]} | "
            f"{values[3]} | {values[4]} |"
        )
    lines.extend(["", f"API-failed frames: {summary['api_failure_count']}.", ""])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    summary = summarize(root)
    write_json(root / "repeat_summary.json", summary)
    (root / "repeat_summary.md").write_text(
        render_markdown(summary), encoding="utf-8"
    )
    print(root / "repeat_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
