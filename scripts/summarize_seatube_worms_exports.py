#!/usr/bin/env python3
"""Summarize WoRMS annotations in one or more local SeaTube exports."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.build_seatube_match_manifest import (
    TaxonResolver,
    load_annotation_rows,
    resolve_taxon,
    strict_row_relative_seconds,
    worms_taxa,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/seatube_metrics_exports.json",
        help="JSON list of named metadata exports.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--offline-taxonomy", action="store_true")
    return parser.parse_args()


def density_bucket(size: int) -> str:
    if size == 1:
        return "one"
    if size <= 4:
        return "few_2_to_4"
    if size <= 9:
        return "many_5_to_9"
    return "many_10_plus"


def taxon_key(taxon: dict[str, Any]) -> tuple[int | None, int | None]:
    try:
        taxonomy_id = int(taxon["taxonomyId"])
    except (KeyError, TypeError, ValueError):
        taxonomy_id = None
    try:
        taxon_id = int(taxon["taxonId"])
    except (KeyError, TypeError, ValueError):
        taxon_id = None
    return taxonomy_id, taxon_id


def summarize_dataset(
    name: str,
    path: Path,
    resolver: TaxonResolver,
) -> tuple[dict[str, Any], Counter[tuple[int | None, int | None]]]:
    rows = load_annotation_rows(path)
    worms_rows = [row for row in rows if worms_taxa(row)]
    mapped_rows = [row for row in worms_rows if row.get("archiveFilename")]
    strict_rows: list[dict[str, Any]] = []
    invalid_mapping_ids: list[int] = []
    for row in mapped_rows:
        try:
            strict_row_relative_seconds(row)
        except (KeyError, TypeError, ValueError):
            if row.get("annotationId") is not None:
                invalid_mapping_ids.append(int(row["annotationId"]))
            continue
        strict_rows.append(row)

    timestamp_groups: dict[tuple[str, str], set[int]] = defaultdict(set)
    taxon_counts: Counter[tuple[int | None, int | None]] = Counter()
    mode_counts: Counter[str] = Counter()
    for row in strict_rows:
        mode_counts[str(row.get("cameraMode") or "unknown")] += 1
        group_key = (str(row["archiveFilename"]), str(row["startDate"]))
        if row.get("annotationId") is not None:
            timestamp_groups[group_key].add(int(row["annotationId"]))
        for taxon in worms_taxa(row):
            taxon_counts[taxon_key(taxon)] += 1

    group_sizes = [len(ids) for ids in timestamp_groups.values()]
    density = Counter(density_bucket(size) for size in group_sizes)
    return (
        {
            "name": name,
            "metadata_file": str(path),
            "export_annotation_rows": len(rows),
            "worms_annotation_rows": len(worms_rows),
            "strictly_mapped_worms_annotations": len(strict_rows),
            "invalid_legacy_mapping_count": len(invalid_mapping_ids),
            "invalid_legacy_mapping_annotation_ids": sorted(invalid_mapping_ids),
            "unique_archive_videos": len(
                {str(row["archiveFilename"]) for row in strict_rows}
            ),
            "unique_timestamp_groups": len(timestamp_groups),
            "max_annotations_at_one_timestamp": max(group_sizes, default=0),
            "timestamp_group_density": dict(sorted(density.items())),
            "camera_mode_annotations": dict(sorted(mode_counts.items())),
            "unique_worms_taxa": len(taxon_counts),
        },
        taxon_counts,
    )


def taxon_rows(
    counts: Counter[tuple[int | None, int | None]], resolver: TaxonResolver
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (taxonomy_id, taxon_id), count in counts.most_common():
        raw = {
            "taxonomyCode": "WoRMS",
            "taxonomyId": taxonomy_id,
            "taxonId": taxon_id,
            "displayText": None,
        }
        resolved = resolve_taxon(raw, resolver)
        rows.append({**resolved, "annotation_mentions": count})
    return rows


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# SeaTube WoRMS Export Metrics",
        "",
        "These counts are computed from the listed local annotation exports. "
        "Only taxonomy ID 1 / taxonomy code WoRMS is included, and video counts "
        "include only strict timestamp-contained mappings.",
        "",
        "## Overall",
        "",
    ]
    overall = payload["overall"]
    for label, key in [
        ("WoRMS annotation rows", "worms_annotation_rows"),
        ("Strictly mapped WoRMS annotations", "strictly_mapped_worms_annotations"),
        ("Unique archive videos", "unique_archive_videos"),
        ("Unique timestamp groups", "unique_timestamp_groups"),
        ("Unique WoRMS taxa", "unique_worms_taxa"),
        ("Invalid legacy mappings", "invalid_legacy_mapping_count"),
    ]:
        lines.append(f"- {label}: {overall[key]:,}")
    lines.extend(
        [
            "",
            "## Per export",
            "",
            "| Export | WoRMS annotations | Strict mappings | Videos | Timestamp groups | Max at one timestamp | Unique taxa |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["datasets"]:
        lines.append(
            f"| {row['name']} | {row['worms_annotation_rows']:,} | "
            f"{row['strictly_mapped_worms_annotations']:,} | "
            f"{row['unique_archive_videos']:,} | {row['unique_timestamp_groups']:,} | "
            f"{row['max_annotations_at_one_timestamp']:,} | {row['unique_worms_taxa']:,} |"
        )
    lines.extend(
        [
            "",
            "## Timestamp density",
            "",
            "| Group size | Timestamp groups |",
            "| --- | ---: |",
        ]
    )
    for key, value in overall["timestamp_group_density"].items():
        lines.append(f"| {key} | {value:,} |")
    lines.extend(
        [
            "",
            "## Most frequent WoRMS taxa",
            "",
            "| Taxon | Rank | ONC taxon ID | AphiaID | Annotation mentions |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in payload["top_taxa"][:30]:
        lines.append(
            f"| {row.get('display_name') or '(unresolved)'} | "
            f"{row.get('rank') or ''} | {row.get('onc_taxon_id') or ''} | "
            f"{row.get('aphia_id') or ''} | {row['annotation_mentions']:,} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    resolver = TaxonResolver(
        output_dir / "onc_taxonomy_cache.json", offline=args.offline_taxonomy
    )
    datasets: list[dict[str, Any]] = []
    overall_taxa: Counter[tuple[int | None, int | None]] = Counter()
    for item in config.get("exports", []):
        path = Path(str(item["metadata_file"])).expanduser().resolve()
        row, taxa = summarize_dataset(str(item["name"]), path, resolver)
        datasets.append(row)
        overall_taxa.update(taxa)

    density: Counter[str] = Counter()
    for row in datasets:
        density.update(row["timestamp_group_density"])
    overall = {
        "worms_annotation_rows": sum(row["worms_annotation_rows"] for row in datasets),
        "strictly_mapped_worms_annotations": sum(
            row["strictly_mapped_worms_annotations"] for row in datasets
        ),
        "unique_archive_videos": sum(row["unique_archive_videos"] for row in datasets),
        "unique_timestamp_groups": sum(row["unique_timestamp_groups"] for row in datasets),
        "unique_worms_taxa": len(overall_taxa),
        "invalid_legacy_mapping_count": sum(
            row["invalid_legacy_mapping_count"] for row in datasets
        ),
        "timestamp_group_density": dict(sorted(density.items())),
    }
    payload = {
        "schema_version": 1,
        "taxonomy_filter": {"taxonomy_code": "WoRMS", "taxonomy_id": 1},
        "datasets": datasets,
        "overall": overall,
        "top_taxa": taxon_rows(overall_taxa, resolver),
    }
    resolver.save()
    write_json(output_dir / "worms_metrics.json", payload)
    (output_dir / "worms_metrics.md").write_text(
        render_markdown(payload), encoding="utf-8"
    )
    print(f"Wrote {output_dir / 'worms_metrics.json'}", flush=True)
    print(f"Wrote {output_dir / 'worms_metrics.md'}", flush=True)


if __name__ == "__main__":
    main()
