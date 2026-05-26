#!/usr/bin/env python3
"""Merge a SoM-augmented JSONL into a fresh copy of frame_outputs_rle.json.

The input frame_outputs file is never mutated; the merged result is
written to --output.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frame-outputs", required=True,
                   help="Original frame_outputs_rle.json (never mutated).")
    p.add_argument("--augmented", required=True,
                   help="Augmented JSONL emitted by run_som_stage.")
    p.add_argument("--output", required=True,
                   help="Where to write the merged JSON.")
    args = p.parse_args(argv)

    with open(args.frame_outputs, "r", encoding="utf-8") as f:
        original = json.load(f)
    augmented_rows: dict[int, dict] = {}
    with open(args.augmented, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            augmented_rows[int(row["frame_index"])] = row

    merged = copy.deepcopy(original)
    new_frames: list[dict] = []
    seen_indices: set[int] = set()
    for row in merged.get("frames", []):
        idx = int(row.get("frame_index", -1))
        seen_indices.add(idx)
        if idx in augmented_rows:
            new_frames.append(augmented_rows[idx])
        else:
            new_frames.append(row)
    # Augmented JSONL may contain frames that weren't in the original
    for idx, row in augmented_rows.items():
        if idx not in seen_indices:
            new_frames.append(row)
    new_frames.sort(key=lambda r: int(r["frame_index"]))
    merged["frames"] = new_frames

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
        f.write("\n")
    print(f"[merge] wrote {args.output} ({len(new_frames)} frames; "
          f"{len(augmented_rows)} augmented)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
