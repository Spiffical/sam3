# SAM3 all-detectable-life prompt diagnosis

_Exploratory diagnostic: 2026-08-17. The direct prompt matrix made no MLLM or
Anthropic API calls._

## Question

The original pipeline asks for `small creatures`. The broader product target is
now all detectable marine life, including motile animals, sessile animals and
colonies, and—when actually present—algae or marine plants.

The five current SeaTube frames are deep-sea scenes. Their coral, gorgonians,
sponges, anemones, tube worms, and crinoids are animals, not plants. Calling
these structures "underwater plants" can be a useful visual analogy, but it is
taxonomically wrong and may make both prompting and evaluation less reliable.

## Controlled experiment

`scripts/probe_sam3_life_prompts.py` bypasses Sonnet and encodes each fixed
frame once. It tested:

- four generic creature phrases;
- five generic life/organism phrases;
- seven plant/vegetation phrases;
- eight sessile morphology phrases; and
- each clip's candidate WoRMS label, expanded into exact scientific and common
  name variants.

The 157 prompt/frame trials produced no runtime errors. Results were recorded at
SAM3 score thresholds 0.20, 0.30, and the pipeline threshold 0.40. Counts are
retrieval outputs, not ground truth.

WSL-only run:

`runs/presentation_benchmark/seatube_meagan_v1/sam3_text_probe/20260817_life_worms_threshold020`

## Results

| Prompt group | Trials | Phrases with any mask at 0.40 | Phrases with any mask at 0.20 |
| --- | ---: | ---: | ---: |
| Generic creature | 20 | 5 | 6 |
| Generic life | 25 | 1 | 2 |
| Plant language | 35 | 0 | 2 |
| Sessile morphology | 40 | 3 | 4 |
| WoRMS taxon variants | 37 | 11 | 16 |

### Plant language does not solve the problem

`plants`, `underwater plants`, `marine plants`, `aquatic plants`, `vegetation`,
`seaweed`, and `algae` returned zero masks at threshold 0.40 on every frame.
At threshold 0.20, only `plants` returned anything: one background sea fan in
the sea-star scene and four anemones in the mixed field. Visual inspection
shows those are animal masks triggered by visual resemblance, not marine-plant
recognition. Lowering the global threshold would therefore add semantic errors
without recovering the dense scenes.

### Generic life language is weaker than `small creatures`

At threshold 0.40, the generic-life group succeeded only for `marine life` on
the isolated crab. `underwater life`, `living organisms`, `marine organisms`,
and `benthic organisms` were ineffective. `small creatures` remains a useful
motile/small-animal proposal prompt on the easy and medium cases, but every
generic phrase returned zero on both dense coral frames even at threshold 0.20.

### Morphology helps only when SAM3 already has the visual concept

On the sea-star/coral frame, `coral` returned six masks and `branching coral`
returned three. The `coral` overlay includes the large colony and background
colonies, but also absorbs or separately returns some sea stars, so it is a
proposal source rather than a final instance result. In the dense thicket and
wide garden, `coral`, `branching coral`, `sea fan`, `gorgonian`, and `sponge`
all returned zero even at threshold 0.20.

### WoRMS vocabulary is useful but inconsistent

Strong or useful cases:

- `Brachyura (crabs)`, `Brachyura`, and `crabs` each retrieved the isolated
  crab; the common plural had the highest score.
- `Enallopsammia rostrata` returned one high-confidence, visually precise mask
  of the central stony-coral colony. The generic `coral` prompt was less
  instance-specific.
- `Actiniaria` returned six masks in the mixed field. Visual review confirms
  multiple good anemone masks, plus some non-anemone/tube-associated objects
  that still require verification.
- `Walteria` returned four plausible slender sponge-like structures in the wide
  garden.

Important failures and caveats:

- Exact `Asteroidea` and `sea stars` prompts returned zero on the clearly
  visible sea stars, while `small creatures` retrieved all three.
- `Chrysogorgia` returned one marginal object at 0.40 that is not visually
  convincing as a genus-specific identification.
- Dense-thicket `Comatulida` and `Parazoanthidae` proposals appeared only below
  0.40 and were visually dubious. No phrase recovered the broad coral colonies.
- A scientific-name hit is a segmentation proposal, not evidence that the
  returned object belongs to that taxon.

## Sonnet 4.6 versus Sonnet 5

The existing three-repeat first-pass artifacts already provide the controlled
model comparison. No new API calls were needed for this diagnosis.

Sonnet 4.6 remained strongly anchored to the original small-creature request.
On the dense frames it explored terms such as `fish`, `crab`, `shrimp`,
`brittle star`, `crinoid`, `invertebrate`, and generic organism phrases. It
occasionally tried `coral`, but every dense-scene SAM3 lookup failed.

Sonnet 5 reasoned more broadly about sessile life and repeatedly tried `coral`,
`sea fan`, `gorgonian`, `octocoral`, `sponge`, `hydroid`, `polyp`, and
`anemone`. It still received zero masks in all three first-pass repeats on both
dense scenes. The direct prompt matrix confirms that those same phrases fail
even when called deterministically and at threshold 0.20.

The primary dense-scene bottleneck is therefore SAM3 text retrieval, not
Sonnet's visual recognition. There are two secondary architecture gaps:

1. The original `small creatures` objective biases Sonnet 4.6 away from
   sessile/colonial animals.
2. Neither first-pass agent receives the available WoRMS candidate vocabulary,
   so it cannot try useful names such as `Enallopsammia rostrata` or `Walteria`.

## Recommended pipeline change

Use **all visible marine life** as the target concept, with explicit categories:

1. motile and small animals;
2. sessile animals and colonies, including corals, gorgonians, sponges,
   anemones, hydroids, and tube worms;
3. algae and true marine plants only when visually or contextually plausible;
4. microbial mats or other living cover when the dataset defines them as a
   target.

Candidate generation should be the union of:

- `small creatures` for its demonstrated small-animal utility;
- a short morphology bank (`coral`, `branching coral`, `sea fan`, `sponge`,
  `anemone`, and relevant common names);
- candidate-constrained WoRMS scientific/common names when SeaTube annotations
  are available; and
- the existing temporal MLLM click-recovery path for anything text retrieval
  misses.

Every taxon- or morphology-prompted mask must remain an unlabeled proposal until
visual/temporal verification and the later annotation-matching stage. Do not
lower the global SAM3 threshold based on this experiment. Do not use `plants`
as a proxy for sessile animals.

The next scored comparison should add this candidate prompt bank before click
recovery, then run at least three repeats and report mean plus or minus standard
deviation. A separate shallow-water set containing actual algae, kelp, or
seagrass is required before making any claim about marine-plant recall.
