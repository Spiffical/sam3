# Fork patches

This is a fork of [facebookresearch/sam3](https://github.com/facebookresearch/sam3),
diverged at `f0399e7`. It carries only the changes SAM3 itself needs to support
downstream click-driven segmentation. The pipeline that uses it lives in a
separate repository.

Keeping this delta small is the point: it should stay rebasable on upstream.

## Functional — no upstream equivalent

**`sam3/model/sam3_video_inference.py`** — adds `add_mask_prompt()` /
`add_tracker_new_mask()`. Injects an externally produced mask into the video
tracker as a prompt, so a mask derived from image-mode clicks can seed video
propagation. Upstream has no path for this.

**`sam3/model/vitdet.py`** — `_apply_rope(q, k, shape)` recomputes `freqs_cis`
when the token grid differs from the pretrained grid, with `rope_interp` scaling
and cls-token handling. This is what makes a non-native `image_size` work at all.

**`sam3/model/sam3_video_predictor.py`** — `image_size` and
`offload_video_to_cpu` on `start_session`, guarded against desyncing the
tracker's fixed resolution; request routing for `add_mask_prompt`;
`start_frame_idx` accepted alongside `start_frame_index`.

## Portability — candidates for upstreaming

**`sam3/model/utils/sam2_utils.py`** — falls back to OpenCV when decord is
unavailable, which it is on several HPC stacks.

**`sam3/model_builder.py`** — resolves the packaged BPE vocab through
`importlib.resources` instead of the deprecated `pkg_resources`.

**`sam3/train/data/sam3_image_dataset.py`** — decord import made optional, with
a clear error only on the codepath that needs it.

**`sam3/model/sam3_video_inference.py`** — tqdm removed from the propagation
loops, which otherwise flood non-interactive logs.

## Agent

**`sam3/agent/agent_core.py`** and **`client_llm.py`** — substantially extended:
tool-call parsing and repair, invalid-tool-state redirects, mask-verdict
extraction with retry, history compaction, output merge/persist, context-budget
retry and image capping.

**`sam3/agent/client_claude.py`** — Anthropic adapter with the same calling
convention as the OpenAI-compatible path, so it can be bound with
`functools.partial` and passed to `agent_inference` unchanged.

**`sam3/agent/proposal_bank.py`** and three prompts
(`system_prompt_underwater_addendum.txt`,
`system_prompt_iterative_checking_underwater_addendum.txt`,
`system_prompt_proposal_verification.txt`) — loaded by `agent_core`.

## Licensing

SAM3 is distributed under the Meta SAM License, which requires derivative works
to be distributed under the same terms. Everything here is either Meta's code or
a modification of it, so it stays under that license. Original work built *on*
SAM3 lives in a separate repository under its own terms.
