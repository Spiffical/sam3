# Presentation benchmark: FathomNet baseline

_Exploratory run: 2026-08-14. No Anthropic API calls._

## Fixed frames

The comparison uses exactly one manually reviewed frame per example video. The
machine-readable source of truth is
`configs/presentation_benchmark_frames.json`. These frames must remain fixed
for the FathomNet, standard SAM3-agent, and custom SAM3-agent slides.

| Clip | Raw frame | Time | Visual reason |
| --- | ---: | ---: | --- |
| chinacreek | 54 | 1.80 s | Multiple differently sized, camouflaged benthic animals |
| rank01 | 238 | 7.93 s | Several fish are visible after the lighting transition |
| rank02 | 260 | 8.67 s | Large fish fully visible with a clean outline |
| rank03 | 100 | 3.33 s | Swimming animal centered above the seafloor |
| rank04 | 140 | 4.67 s | Highly camouflaged elongated fish on the left seafloor |

The choice came from visual review of 16 evenly spaced candidates per clip,
followed by full-resolution review. Rank01 frame 238 replaced the inherited
darker candidates because it shows the group of fish much more clearly.

## Models and run

Two current official FathomNet detectors were compared:

- [2025 MBARI Benthic Supercategory detector](https://huggingface.co/FathomNet/2025-MBARI-Benthic-Supercategory-Object-Detector)
- [2025 MBARI Midwater Supercategory detector](https://huggingface.co/FathomNet/2025-MBARI-Midwater-Supercategory-Object-Detector)

Both are YOLO11x object detectors trained at 640-pixel input size. The run
tested 640 and 1280 pixels, retained raw detections down to confidence 0.01,
and rendered global confidence thresholds 0.05, 0.10, and 0.25 with
class-agnostic NMS. The WSL-only run is:

`runs/presentation_benchmark/fathomnet/20260814_fathomnet_exploratory_v1/`

It contains the five extracted frames, 60 overlays, 12 visual-review
galleries, raw detections, weight hashes, and source metadata. It is marked
exploratory rather than scored because the inherited source tree is not yet at
a clean baseline commit (`6815d08`, dirty).

## Visual QA

The best single recall-oriented FathomNet baseline is the **benthic model at
640 pixels and confidence 0.05**.

- chinacreek: matches 3 of 5 existing hand-reviewed SAM3 boxes at box IoU at
  least 0.3, with four total predictions. It still misses visible animals.
- rank01: covers several fish, but also produces substrate false positives and
  does not cleanly separate every visible fish.
- rank02: one tight, correct box on the fish.
- rank03: correct box on the central animal plus one obvious false positive.
- rank04: the single box is a false positive on lower-left substrate/shell; the
  camouflaged fish itself is missed.

The midwater model at its conventional 0.25 threshold is cleaner on rank02 and
rank03, but also misses rank04, merges multiple rank01 fish into one box, and
only matches 1 of the 5 existing chinacreek boxes. Neither FathomNet model
detects the actual rank04 fish. Increasing input size to 1280 does not provide
a consistent gain and creates duplicates or confident false positives on
several frames.

Taxonomic predictions are visibly unreliable under this domain shift and are
not part of this comparison. Presentation overlays should therefore use
numbered boxes without class names. This keeps the slide focused on creature
recall and instance separation.

## Additional July 6 FathomNet repositories

Two more official repositories were tested after the first comparison. Both
were last updated on Hugging Face on **2026-07-06**, although that update date
is distinct from the model's training vintage:

- [MBARI-315k YOLOv8](https://huggingface.co/FathomNet/MBARI-315k-yolov8),
  a 499-class fine-grained detector. Repository revision
  `f4a839a78214441e32bee1928839fa7578c2bcd8`.
- [Megalodon 2023 YOLOv8](https://huggingface.co/FathomNet/megalodon-2023-yolov8),
  a YOLOv8x detector trained on all publicly available FathomNet
  localizations with one generic `object` class. Repository revision
  `f312604d2f59dcccde74499cf0e0ae0647331f01`.

The same five fixed frames, resolutions, confidence sweep, and class-agnostic
NMS were used. No Anthropic API calls were made.

### MBARI-315k visual QA

Best global presentation setting: **640 pixels, confidence 0.05**.

- chinacreek: 3/5 existing hand-reviewed boxes matched, with four predictions.
- rank01: only one low-confidence proposal; most of the fish group is missed.
- rank02: one tight, correct fish box.
- rank03: correct central box plus one clear marine-snow false positive.
- rank04: both boxes are false positives; the camouflaged fish is missed.

At 1280 pixels it finds more rank01 fish, but creates duplicate boxes on
rank02 and additional false positives on rank03 without finding rank04. The
WSL exploratory and presentation runs are:

- `runs/presentation_benchmark/fathomnet/20260815_mbari315k_exploratory_v1/`
- `runs/presentation_benchmark/fathomnet/20260815_mbari315k_presentation_v1/`

### Megalodon visual QA

Best global presentation setting: **640 pixels, confidence 0.10**.

- chinacreek: 4/5 existing hand-reviewed boxes matched, with five predictions.
- rank01: three proposals cover only part of the fish group and merge nearby
  fish rather than separating every instance.
- rank02: one tight, correct fish box.
- rank03: the central animal is detected, alongside four false positives on
  water-column particles or substrate.
- rank04: the only retained proposal is a false positive; the camouflaged fish
  is missed.

At 1280 pixels and confidence 0.10, Megalodon reaches 5/5 on the existing
chinacreek boxes, but produces 9 boxes on rank01, 4 overlapping boxes on the
single rank02 fish, 11 boxes on rank03, and 2 false positives on rank04. The
recall gain is therefore not a usable global presentation configuration. The
WSL exploratory and presentation runs are:

- `runs/presentation_benchmark/fathomnet/20260815_megalodon_exploratory_v1/`
- `runs/presentation_benchmark/fathomnet/20260815_megalodon_presentation_v1/`
