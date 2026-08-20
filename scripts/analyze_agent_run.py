"""Analyze a SAM3 agent every-frame video run and (optionally) compare against
a baseline run.

Reports tool-use telemetry per frame: segment_phrase calls and phrases used,
drop_masks usage, examine_each_mask usage, mask counts, error rate, runtime.
Useful for diagnosing whether the agent loop actually exercises the mask
accumulation / verification tools across a video.

Usage:
    python scripts/analyze_agent_run.py <run_dir> [--baseline-summary PATH]
                                         [--baseline-frame-outputs PATH]

`run_dir` is expected to contain:
    summary.json
    frame_results.jsonl  (written by run_sam3_agent_every_frame_video.py)
    frame_outputs_rle.json
    agent_frames/frame_NNNNNN/agent_debug_out/<frame>/debug_history.json
       (only when --debug was used)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


_TOOL_RE = re.compile(r"<tool>\s*(\{.*?\})\s*</tool>", flags=re.DOTALL)


def _safe_load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _parse_tools_from_history(history_path: Path) -> dict[str, Any]:
    """
    Parse a per-frame agent_debug_out debug_history.json (a JSONL of messages)
    and return a tool-use summary.
    """
    tools_in_order: list[str] = []
    segment_phrases: list[str] = []
    examine_calls = 0
    drop_calls = 0
    select_calls = 0
    report_no_mask_calls = 0

    # debug_history.json is written via `f.write(json.dumps(msg, indent=4) + "\n")`
    # so it's not strict JSONL — load as one big concatenated stream by scanning
    # for top-level objects.
    try:
        text = history_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"tools_in_order": [], "segment_phrases": []}
    decoder = json.JSONDecoder()
    pos = 0
    parsed_msgs: list[dict[str, Any]] = []
    while pos < len(text):
        try:
            obj, end = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            pos += 1
            continue
        parsed_msgs.append(obj)
        pos = end
        while pos < len(text) and text[pos] in " \t\r\n":
            pos += 1

    for msg in parsed_msgs:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content") or []
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "text":
                continue
            for tool_match in _TOOL_RE.finditer(item.get("text", "")):
                try:
                    parsed = json.loads(tool_match.group(1))
                except json.JSONDecodeError:
                    continue
                name = parsed.get("name")
                tools_in_order.append(name)
                if name == "segment_phrase":
                    phrase = parsed.get("parameters", {}).get("text_prompt")
                    if isinstance(phrase, str):
                        segment_phrases.append(phrase)
                elif name == "examine_each_mask":
                    examine_calls += 1
                elif name == "drop_masks":
                    drop_calls += 1
                elif name == "select_masks_and_return":
                    select_calls += 1
                elif name == "report_no_mask":
                    report_no_mask_calls += 1

    return {
        "tools_in_order": tools_in_order,
        "segment_phrases": segment_phrases,
        "num_segment_phrase": len(segment_phrases),
        "num_examine_each_mask": examine_calls,
        "num_drop_masks": drop_calls,
        "num_select_masks_and_return": select_calls,
        "num_report_no_mask": report_no_mask_calls,
    }


def _find_history_for_frame(run_dir: Path, frame_index: int) -> Path | None:
    candidates = [
        run_dir
        / "agent_frames"
        / f"frame_{frame_index:06d}"
        / "agent_debug_out",
    ]
    for candidate_root in candidates:
        if not candidate_root.exists():
            continue
        for child in candidate_root.iterdir():
            history_path = child / "debug_history.json"
            if history_path.exists():
                return history_path
    return None


def _count_masks_per_frame_outputs(payload: Any) -> dict[int, int]:
    """Map frame_index -> mask count from a frame_outputs_rle.json payload."""
    if not isinstance(payload, dict):
        return {}
    counts: dict[int, int] = {}
    for row in payload.get("frames", []) or []:
        if not isinstance(row, dict):
            continue
        idx = int(row.get("frame_index", -1))
        masks = row.get("out_binary_masks_rle") or []
        if isinstance(masks, list):
            counts[idx] = len(masks)
    return counts


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "median": median(values),
        "min": min(values),
        "p90": _percentile(values, 0.90),
        "max": max(values),
    }


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.2f}"
    return str(x)


def analyze_run(run_dir: Path) -> dict[str, Any]:
    summary = _safe_load_json(run_dir / "summary.json") or {}
    frame_results = list(_iter_jsonl(run_dir / "frame_results.jsonl"))
    frame_outputs = _safe_load_json(run_dir / "frame_outputs_rle.json") or {}
    mask_counts_by_frame = _count_masks_per_frame_outputs(frame_outputs)

    per_frame_tool_use: list[dict[str, Any]] = []
    phrase_counter: Counter[str] = Counter()
    tools_seq_counter: Counter[str] = Counter()
    total_segment_calls = 0
    total_examine_calls = 0
    total_drop_calls = 0
    total_select_calls = 0
    total_report_no_mask = 0
    frames_with_multi_segment = 0
    frames_with_drop = 0
    frames_with_examine = 0
    frames_with_report_no_mask = 0
    err_frames = 0
    skipped_frames = 0
    runtimes: list[float] = []
    history_len_per_frame: list[int] = []
    mask_counts_per_analyzed_frame: list[int] = []

    for row in frame_results:
        idx = int(row.get("frame_index", -1))
        if row.get("skipped"):
            skipped_frames += 1
            continue
        if row.get("error"):
            err_frames += 1
        rt = row.get("frame_runtime_sec")
        if isinstance(rt, (int, float)):
            runtimes.append(float(rt))
        history_len = row.get("history_len")
        if isinstance(history_len, int):
            history_len_per_frame.append(history_len)

        num_masks = row.get("num_masks")
        if isinstance(num_masks, int):
            mask_counts_per_analyzed_frame.append(num_masks)

        history_path = _find_history_for_frame(run_dir, idx)
        if history_path is not None:
            tool_info = _parse_tools_from_history(history_path)
        else:
            tool_info = {
                "tools_in_order": [],
                "segment_phrases": list(row.get("segment_prompts") or []),
                "num_segment_phrase": len(row.get("segment_prompts") or []),
                "num_examine_each_mask": 0,
                "num_drop_masks": 0,
                "num_select_masks_and_return": 0,
                "num_report_no_mask": 0,
            }
        per_frame_tool_use.append(
            {
                "frame_index": idx,
                "num_masks": num_masks,
                "frame_runtime_sec": rt,
                "history_len": history_len,
                **tool_info,
            }
        )
        total_segment_calls += tool_info.get("num_segment_phrase", 0)
        total_examine_calls += tool_info.get("num_examine_each_mask", 0)
        total_drop_calls += tool_info.get("num_drop_masks", 0)
        total_select_calls += tool_info.get("num_select_masks_and_return", 0)
        total_report_no_mask += tool_info.get("num_report_no_mask", 0)
        if tool_info.get("num_segment_phrase", 0) >= 2:
            frames_with_multi_segment += 1
        if tool_info.get("num_drop_masks", 0) >= 1:
            frames_with_drop += 1
        if tool_info.get("num_examine_each_mask", 0) >= 1:
            frames_with_examine += 1
        if tool_info.get("num_report_no_mask", 0) >= 1:
            frames_with_report_no_mask += 1
        for phrase in tool_info.get("segment_phrases", []):
            phrase_counter[phrase] += 1
        for tool_name in tool_info.get("tools_in_order", []):
            tools_seq_counter[tool_name] += 1

    analyzed = max(1, len(per_frame_tool_use))
    return {
        "run_dir": str(run_dir),
        "summary_block": summary,
        "frames_total": len(frame_results),
        "frames_analyzed": len(per_frame_tool_use),
        "frames_skipped": skipped_frames,
        "frames_with_error": err_frames,
        "frames_with_multi_segment_phrase": frames_with_multi_segment,
        "frames_with_drop_masks": frames_with_drop,
        "frames_with_examine_each_mask": frames_with_examine,
        "frames_with_report_no_mask": frames_with_report_no_mask,
        "tool_call_counts": {
            "segment_phrase": total_segment_calls,
            "examine_each_mask": total_examine_calls,
            "drop_masks": total_drop_calls,
            "select_masks_and_return": total_select_calls,
            "report_no_mask": total_report_no_mask,
        },
        "mask_count_stats": _stats(mask_counts_per_analyzed_frame),
        "runtime_sec_stats": _stats(runtimes),
        "history_len_stats": _stats([float(x) for x in history_len_per_frame]),
        "phrase_counter": dict(phrase_counter.most_common(20)),
        "tools_seq_counter": dict(tools_seq_counter),
        "mask_counts_by_frame": mask_counts_by_frame,
        "per_frame_tool_use": per_frame_tool_use,
    }


def _print_block(label: str, report: dict[str, Any], frames_to_compare: int | None) -> None:
    summary = report["summary_block"]
    print(f"\n=== {label} ===")
    print(f"  run_dir: {report['run_dir']}")
    if frames_to_compare is not None:
        print(f"  comparing on first {frames_to_compare} frames")
    print(f"  model: {summary.get('model') or summary.get('claude_model') or 'unknown'}")
    print(
        "  frames: total={total} analyzed={ana} skipped={sk} errors={err}".format(
            total=report["frames_total"],
            ana=report["frames_analyzed"],
            sk=report["frames_skipped"],
            err=report["frames_with_error"],
        )
    )
    print("  tool call totals:")
    for k, v in report["tool_call_counts"].items():
        print(f"    {k:30s} {v}")
    print(
        "  frames with extra tools: multi_segment={ms} drop_masks={dm} examine={ex} report_no_mask={rn}".format(
            ms=report["frames_with_multi_segment_phrase"],
            dm=report["frames_with_drop_masks"],
            ex=report["frames_with_examine_each_mask"],
            rn=report["frames_with_report_no_mask"],
        )
    )
    print(f"  mask count per frame: {report['mask_count_stats']}")
    print(f"  frame runtime sec:    {report['runtime_sec_stats']}")
    print(f"  agent history len:    {report['history_len_stats']}")
    print(f"  top segment_phrase phrases (count): {report['phrase_counter']}")


def _compare_mask_counts(claude: dict[int, int], baseline: dict[int, int], n: int) -> None:
    print("\n=== Per-frame mask count diff (Claude vs baseline) ===")
    rows = []
    common = sorted(set(claude.keys()) & set(baseline.keys()))[:n]
    if not common:
        print("  (no overlapping frames found between Claude and baseline frame_outputs.)")
        return
    diffs = []
    for idx in common:
        c = claude.get(idx, 0)
        b = baseline.get(idx, 0)
        diffs.append(c - b)
        rows.append((idx, c, b, c - b))
    print(f"  frames compared: {len(common)}")
    print(f"  claude total masks:   {sum(c for _, c, _, _ in rows)}")
    print(f"  baseline total masks: {sum(b for _, _, b, _ in rows)}")
    print(f"  mean diff (claude - baseline): {sum(diffs) / len(diffs):+.2f}")
    print(f"  median diff:                   {median(diffs):+.2f}")
    print(f"  frames where claude has more:  {sum(1 for d in diffs if d > 0)}")
    print(f"  frames where baseline has more:{sum(1 for d in diffs if d < 0)}")
    # print a few sample rows
    print("  first 10 rows (idx, claude_n, baseline_n, diff):")
    for r in rows[:10]:
        print("   ", r)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="Path to the agent run directory.")
    ap.add_argument(
        "--baseline-summary",
        default="",
        help="Optional baseline summary.json (e.g. Qwen Nibi run) for side-by-side metrics.",
    )
    ap.add_argument(
        "--baseline-frame-outputs",
        default="",
        help="Optional baseline frame_outputs_rle.json for per-frame mask count diff.",
    )
    ap.add_argument(
        "--baseline-frame-results",
        default="",
        help="Optional baseline frame_results.jsonl for tool-use comparison.",
    )
    ap.add_argument(
        "--frames-to-compare",
        type=int,
        default=0,
        help="Limit baseline comparison to first N frames. 0 = all available.",
    )
    args = ap.parse_args()

    claude_report = analyze_run(Path(args.run_dir))
    n_compare = args.frames_to_compare or claude_report["frames_analyzed"]
    _print_block("Claude (or current) run", claude_report, frames_to_compare=None)

    if args.baseline_summary or args.baseline_frame_outputs or args.baseline_frame_results:
        print("\n--- Baseline ---")
        if args.baseline_summary:
            baseline_summary = _safe_load_json(Path(args.baseline_summary)) or {}
            print(f"baseline summary.json: {args.baseline_summary}")
            for k in (
                "model",
                "processed_frames",
                "analyzed_frames",
                "frames_with_masks",
                "total_masks",
                "avg_masks_per_frame",
                "error_count",
                "runtime_sec",
                "throughput_fps",
            ):
                if k in baseline_summary:
                    print(f"  {k}: {_fmt(baseline_summary[k])}")
        if args.baseline_frame_outputs:
            baseline_payload = _safe_load_json(Path(args.baseline_frame_outputs))
            baseline_counts = _count_masks_per_frame_outputs(baseline_payload)
            print(f"baseline frame_outputs: {args.baseline_frame_outputs}")
            print(f"  frames in baseline: {len(baseline_counts)}")
            _compare_mask_counts(
                claude_report["mask_counts_by_frame"], baseline_counts, n_compare
            )
        if args.baseline_frame_results:
            baseline_results = list(_iter_jsonl(Path(args.baseline_frame_results)))
            print(f"\nbaseline frame_results.jsonl: {args.baseline_frame_results}")
            phrases = Counter()
            seg_calls = 0
            for row in baseline_results[:n_compare]:
                for p in row.get("segment_prompts") or []:
                    phrases[p] += 1
                    seg_calls += 1
            print(f"  baseline total segment_phrase calls (first {n_compare}): {seg_calls}")
            print(f"  baseline top phrases: {dict(phrases.most_common(10))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
