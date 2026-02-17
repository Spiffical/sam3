#!/usr/bin/env python3
"""
Generate a copy-paste report for sending results back to Codex.
"""

import argparse
import csv
import os
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    return parser.parse_args()


def read_rows(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def format_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "model_key",
        "status",
        "runtime_sec",
        "num_agent_masks",
        "num_generated_prompts",
        "frames_with_outputs",
        "frame_output_fraction",
    ]
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(row.get(h, "") for h in headers) + " |")
    return "\n".join(out)


def build_report(rows: list[dict[str, Any]]) -> str:
    video_path = rows[0].get("video_path", "") if rows else ""
    table = format_table(rows) if rows else "No rows found in summary.csv"
    return f"""# Nibi Model Comparison Report (Paste To Codex)

## Run Context

- Video path: `{video_path}`
- Objective: segment and track marine life in ONC footage using SAM3 agent mode.
- Comparison scope: Gemini baseline vs open models (Qwen/Kimi).

## Quantitative Summary

{table}

## Qualitative Notes (Fill In)

- Which model produced the most biologically plausible masks?
- Which model missed obvious organisms?
- Which model over-segmented background noise (sediment, lighting artifacts)?
- Which model had the best mask continuity after propagation?

## Compute Notes (Fill In)

- Queue wait time per run:
- GPU allocation per run:
- Peak memory per run (if captured):
- Any OOM or startup failures:

## Questions For Codex

1. Based on this table and notes, which model should be the scale-up candidate?
2. What changes should we make before launching a multi-video array?
3. Should we use stricter prompts, frame subsampling, or prompt ensembling?
"""


def main() -> int:
    args = parse_args()
    rows = read_rows(args.summary_csv)
    report = build_report(rows)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as handle:
        handle.write(report)
    print(f"Wrote {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
