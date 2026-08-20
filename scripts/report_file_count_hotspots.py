#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DirStats:
    path: Path
    depth: int
    direct_files: int = 0
    direct_dirs: int = 0
    total_files: int = 0
    total_dirs: int = 0
    errors: int = 0
    extension_counts: Counter[str] = field(default_factory=Counter)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report which subdirectories contribute the most files under a root path. "
            "Useful for diagnosing scratch quota issues caused by file count."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Root directory to analyze. Default: current directory",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help=(
            "Show recursive totals for directories up to this depth below the root. "
            "Default: 3"
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=25,
        help="Number of rows to show in each ranking. Default: 25",
    )
    parser.add_argument(
        "--leaf-top",
        type=int,
        default=25,
        help="Number of leaf directories to show. Default: 25",
    )
    parser.add_argument(
        "--extensions-top",
        type=int,
        default=20,
        help="Number of file extensions to show. Default: 20",
    )
    parser.add_argument(
        "--min-files",
        type=int,
        default=1,
        help="Hide directories with fewer recursive files than this. Default: 1",
    )
    parser.add_argument(
        "--follow-symlinks",
        action="store_true",
        help="Follow symlinked directories. Default: off",
    )
    return parser.parse_args()


def normalize_ext(path: Path) -> str:
    suffix = path.suffix.lower()
    return suffix if suffix else "[no_ext]"


def walk_tree(
    path: Path,
    *,
    depth: int,
    root_depth: int,
    max_depth: int,
    follow_symlinks: bool,
    tracked: list[DirStats],
    leaf_dirs: list[DirStats],
    global_exts: Counter[str],
) -> DirStats:
    stats = DirStats(path=path, depth=depth - root_depth)
    include_in_rankings = stats.depth <= max_depth

    if include_in_rankings:
        tracked.append(stats)

    child_dir_count = 0
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                try:
                    if entry.is_file(follow_symlinks=follow_symlinks):
                        stats.direct_files += 1
                        stats.total_files += 1
                        ext = normalize_ext(Path(entry.name))
                        stats.extension_counts[ext] += 1
                        global_exts[ext] += 1
                    elif entry.is_dir(follow_symlinks=follow_symlinks):
                        child_dir_count += 1
                        stats.direct_dirs += 1
                        child_stats = walk_tree(
                            Path(entry.path),
                            depth=depth + 1,
                            root_depth=root_depth,
                            max_depth=max_depth,
                            follow_symlinks=follow_symlinks,
                            tracked=tracked,
                            leaf_dirs=leaf_dirs,
                            global_exts=global_exts,
                        )
                        stats.total_files += child_stats.total_files
                        stats.total_dirs += 1 + child_stats.total_dirs
                        stats.errors += child_stats.errors
                        stats.extension_counts.update(child_stats.extension_counts)
                except (PermissionError, FileNotFoundError, OSError):
                    stats.errors += 1
    except (PermissionError, FileNotFoundError, NotADirectoryError, OSError):
        stats.errors += 1
        return stats

    if child_dir_count == 0:
        leaf_dirs.append(stats)
    return stats


def format_int(value: int) -> str:
    return f"{value:,}"


def format_rel(path: Path, root: Path) -> str:
    if path == root:
        return "."
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def print_table(title: str, rows: list[DirStats], root: Path, limit: int) -> None:
    print(title)
    if not rows:
        print("  none")
        print()
        return

    shown = rows[:limit]
    path_width = max(len(format_rel(row.path, root)) for row in shown)
    path_width = min(max(path_width, 4), 110)
    header = (
        f"  {'files(rec)':>12}  {'files(dir)':>10}  {'dirs(rec)':>10}  {'errs':>6}  path"
    )
    print(header)
    for row in shown:
        rel = format_rel(row.path, root)
        if len(rel) > path_width:
            rel = "..." + rel[-(path_width - 3) :]
        print(
            f"  {format_int(row.total_files):>12}  "
            f"{format_int(row.direct_files):>10}  "
            f"{format_int(row.total_dirs):>10}  "
            f"{format_int(row.errors):>6}  {rel}"
        )
    print()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"Root does not exist: {root}", file=sys.stderr)
        return 1
    if not root.is_dir():
        print(f"Root is not a directory: {root}", file=sys.stderr)
        return 1
    if args.max_depth < 0:
        print("--max-depth must be >= 0", file=sys.stderr)
        return 1

    tracked: list[DirStats] = []
    leaf_dirs: list[DirStats] = []
    global_exts: Counter[str] = Counter()
    root_depth = len(root.parts)

    root_stats = walk_tree(
        root,
        depth=root_depth,
        root_depth=root_depth,
        max_depth=args.max_depth,
        follow_symlinks=args.follow_symlinks,
        tracked=tracked,
        leaf_dirs=leaf_dirs,
        global_exts=global_exts,
    )

    min_files = max(0, int(args.min_files))
    ranked = [row for row in tracked if row.total_files >= min_files]
    ranked.sort(key=lambda row: (-row.total_files, str(row.path)))

    direct_ranked = [row for row in ranked if row.direct_files > 0]
    direct_ranked.sort(key=lambda row: (-row.direct_files, str(row.path)))

    leaf_ranked = [row for row in leaf_dirs if row.total_files >= min_files]
    leaf_ranked.sort(key=lambda row: (-row.total_files, str(row.path)))

    print(f"Root: {root}")
    print(f"Total files: {format_int(root_stats.total_files)}")
    print(f"Total directories: {format_int(root_stats.total_dirs)}")
    print(f"Traversal errors: {format_int(root_stats.errors)}")
    print()

    print_table(
        f"Top recursive file-count hotspots up to depth {args.max_depth}",
        ranked,
        root,
        args.top,
    )
    print_table(
        "Top direct file-count directories",
        direct_ranked,
        root,
        args.top,
    )
    print_table(
        "Top leaf directories by file count",
        leaf_ranked,
        root,
        args.leaf_top,
    )

    print(f"Top file extensions ({args.extensions_top})")
    if not global_exts:
        print("  none")
        return 0
    for ext, count in global_exts.most_common(args.extensions_top):
        print(f"  {format_int(count):>12}  {ext}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
