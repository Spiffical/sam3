# SeaTube WoRMS object-matching foundation

## Scope

The first annotation-matching experiments use only WoRMS annotations:

- `taxonomyId == 1`, or equivalently `taxonomyCode == "WoRMS"`
- ONC `taxonId` values are internal ONC identifiers, not WoRMS AphiaIDs
- names and external identifiers are resolved through
  `/internal/taxonomies/{taxonomyId}/taxons/{taxonId}`

No taxonomy is inferred from pixels in this stage. SeaTube annotations remain
candidate labels until a later visual matcher associates them with numbered
segmented objects.

## Inventory evidence

The existing full-history SeaTube inventory generated on 2026-02-22 reports:

- 579,181 annotations overall
- 214,106 WoRMS annotations (36.97%)
- 23,383 archive clips containing WoRMS annotations
- 2,707.97 archive-video hours containing WoRMS annotations

That report is a historical API inventory through 2026-02-22, not a live count.

The reproducible local-export metrics pass over Dive 1473, Dive 6370, and
Stationary 2335 reports:

- 8,849 WoRMS annotation rows
- 8,708 strict timestamp-contained video mappings
- 158 unique archive videos
- 3,665 exact video/timestamp groups
- 104 unique WoRMS taxa
- 141 invalid legacy mappings, all from Stationary 2335

The invalid mappings are the consequence of the former nearest-video fallback:
annotations in recording gaps could be attached to another clip (even another
day), after which negative times were clamped to zero. The downloader now
requires strict containment and the window analyzers reject legacy rows outside
the mapped clip.

### Live annotator coverage scan

The WSL full-history scan completed on 2026-08-17 UTC and retained 264,969
WoRMS observations from 184 creators. It scanned 2,058 dive records and 39
stationary locations at low-video resolution. The report is stored only on WSL
under `downloads/sam3_seatube_matching_v1/annotator_leaderboard_full_20260816`.

| Coverage rank | Creator | Annotations | Strict videos | Scopes | Taxa | Strict mapping | Reviewed signal | Cross-creator collision groups |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Meagan Putts | 41,383 | 4,393 | 87 | 820 | 98.56% | 86.07% | 83 |
| 2 | Sarah Bingo | 40,725 | 3,995 | 79 | 838 | 96.36% | 88.84% | 126 |
| 3 | Deep-sea Animal Research Center (DARC) | 51,950 | 3,750 | 45 | 911 | 99.90% | 100.00% | 11 |
| 4 | Upasana Ganguly | 9,582 | 3,038 | 104 | 616 | 99.78% | 98.51% | 15 |
| 5 | Lauren Walling | 5,867 | 2,345 | 64 | 428 | 100.00% | 99.68% | 3 |
| 6 | Calder Guimond | 7,644 | 1,618 | 59 | 124 | 99.76% | 99.97% | 2 |

Use Meagan Putts (`userId=44113`) as the initial broad-coverage human cohort.
Keep DARC (`userId=283600`) as a separate institutional/high-review comparison,
and Calder Guimond (`userId=145400`) as a smaller, exceptionally clean human
comparison. Never pool these cohorts into one score.

This is a high-coverage inventory, not a claim of perfect exhaustiveness: 30 of
2,058 dive annotation fetches failed after the API call, and 12 stationary nodes
listed in ONC's tree were rejected by ONC's own video endpoint. Both conditions
are recorded in the report. Per-camera-mode observation checkpoints make later
reruns resumable and prevent a stationary metadata failure from discarding a
completed dive scan.

## Fixed one/few/many samples

`configs/seatube_exact_frame_samples.json` is the fixed experiment definition.

| Sample | Difficulty | Annotation candidates | Visual QA |
| --- | --- | ---: | --- |
| `seatube_one_rockfish` | one | 1 from Sofia Jimenez Gonzalez | Excellent: one large, crisp fish; exact timestamp |
| `seatube_few_anemone_stars` | few | 1 from Calder Guimond | Hard case: pink anemone plus small sea stars; the stars intentionally remain unmatched rather than mixing annotators |
| `seatube_many_pelagic` | many | 13 from Ashley Marranzino | Strong density case: many visible pelagic animals; all 13 annotations at the exact timestamp |

The “many” frame was selected after visual comparison with four high-density
alternatives. The initially highest-count frame had 14 annotations but was too
ambiguous against marine snow for a clear presentation.

## Annotator cohort policy

Each manifest is built for exactly one configured `createdBy.userId`. Mixing
annotators within a sample is forbidden, and the expected candidate count is
asserted so later export changes fail loudly. Names are retained for provenance;
email addresses are never copied into manifests.

Same-creator annotations at the same timestamp are preserved. They can represent
distinct observations or carry count metadata and therefore are not safe to
deduplicate automatically. Cross-creator collisions are reported by the
leaderboard analyzer, but one creator is selected before matching. Scored
results should be reported per creator cohort rather than pooling annotators.
An annotation-event count is consequently not a ground-truth creature count;
visible-object recall must still come from the segmentation review.

The fixed “few” frame previously mixed a sea-star annotation from Damian Rohraff
with an anemone annotation from Calder Guimond. Calder has the broader local
Dive 6370 coverage, so the benchmark now uses only Calder's exact-frame anemone
candidate. The visible sea stars are a useful negative control: a matcher must
leave them unmatched instead of borrowing a label from another cohort.

## Reproducible commands

All generated frames, caches, manifests, and reports stay on WSL:

```bash
scripts/gpu_python.sh scripts/summarize_seatube_worms_exports.py \
  --config configs/seatube_metrics_exports.json \
  --output-dir /home/sbialek/ONC/seatube-downloader/downloads/sam3_seatube_matching_v1/metrics

scripts/gpu_python.sh scripts/build_seatube_match_manifest.py \
  --config configs/seatube_exact_frame_samples.json \
  --output-dir /home/sbialek/ONC/seatube-downloader/downloads/sam3_seatube_matching_v1/samples
```

These commands make no Claude API calls. Each sample output contains:

- the exact decoded frame and a one-frame MP4 for the SAM3 first pass
- timestamp deltas for every WoRMS annotation candidate
- resolved ONC/WoRMS taxon metadata
- a matching manifest with explicit unmatched annotation/object fields
- optionally, numbered overlays and object crops when a segmentation root is supplied

## Next matching stage

1. Run the settled SAM3 agent/custom click flow on the three one-frame MP4s,
   using the full source video for temporal context in the missed-creature pass.
2. Re-run `build_seatube_match_manifest.py --segmentation-root <run>` to attach
   numbered masks, bounding boxes, and per-object crops.
3. Give Claude the full numbered frame, object crops, annotation candidates,
   comments, exact time deltas, and nearby video evidence.
4. Require structured mappings plus explicit `unmatched_object_ids` and
   `unmatched_annotation_ids`. Never force a complete assignment.
5. Permit one annotation to match multiple objects when the annotation is a
   presence-level taxon tag and several same-taxon individuals are visible.
