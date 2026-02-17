#!/usr/bin/env python3
"""
Summarize per-model outputs into CSV and Markdown tables.
"""

import argparse
import csv
import json
import os
import time
from typing import Any


SUMMARY_FIELDS = [
    "model_key",
    "model",
    "status",
    "runtime_sec",
    "num_agent_masks",
    "num_generated_prompts",
    "frames_with_outputs",
    "total_video_frames",
    "frame_output_fraction",
    "video_path",
    "output_video_path",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True, type=str)
    parser.add_argument("--summary_csv", default=None, type=str)
    parser.add_argument("--summary_md", default=None, type=str)
    return parser.parse_args()


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_runs(output_root: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not os.path.isdir(output_root):
        return rows

    for name in sorted(os.listdir(output_root)):
        run_dir = os.path.join(output_root, name)
        if not os.path.isdir(run_dir):
            continue

        run_metrics_path = os.path.join(run_dir, "run_metrics.json")
        launcher_metrics_path = os.path.join(run_dir, "launcher_metrics.json")

        row: dict[str, Any] = {"model_key": name}
        if os.path.exists(run_metrics_path):
            row.update(load_json(run_metrics_path))
        if os.path.exists(launcher_metrics_path):
            launcher = load_json(launcher_metrics_path)
            row.setdefault("status", launcher.get("status"))
            row.setdefault("model", launcher.get("model_id"))
            row.setdefault("runtime_sec", launcher.get("runtime_sec"))
            row.setdefault("error", launcher.get("error", ""))

        if len(row.keys()) > 1:
            rows.append(row)
    return rows


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in SUMMARY_FIELDS})


def render_markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "model_key",
        "status",
        "runtime_sec",
        "num_agent_masks",
        "num_generated_prompts",
        "frames_with_outputs",
        "frame_output_fraction",
    ]
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines = [header_line, sep_line]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(str(row.get(header, "")) for header in headers)
            + " |"
        )
    return "\n".join(lines)


def write_markdown(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Model Comparison Summary\n\n")
        handle.write(f"Generated at: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n\n")
        if not rows:
            handle.write("No runs found.\n")
            return
        handle.write(render_markdown_table(rows))
        handle.write("\n")


def main() -> int:
    args = parse_args()
    rows = discover_runs(args.output_root)
    rows = sorted(rows, key=lambda row: row.get("model_key", ""))

    summary_csv = args.summary_csv or os.path.join(args.output_root, "summary.csv")
    summary_md = args.summary_md or os.path.join(args.output_root, "summary.md")

    write_csv(summary_csv, rows)
    write_markdown(summary_md, rows)

    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
