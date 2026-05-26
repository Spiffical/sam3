"""Set-of-Mark missed-creature discovery stage.

Given an existing per-frame text-agent run, on a small set of selected
target frames: produce dense SAM3 candidates, drop the ones already
covered, overlay numbered marks, ask the MLLM (with a few unmarked
reference frames as context) which marks are real biological subjects,
and write the accepted masks back into the per-frame outputs.

The MLLM never emits raw (x, y) coordinates -- it selects by mark id.
This sidesteps the failure mode that killed the prior point-proposal
loop (commit ae72a4c).

Spec: docs/superpowers/specs/2026-05-26-som-missed-creature-loop-design.md
"""

from __future__ import annotations

import json
import re

_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


def parse_som_response(text: str, *, valid_ids: set[int] | None = None) -> list[int]:
    """Extract accepted mark ids from an MLLM response.

    Contract: the response is expected to contain a trailing
    ``<answer>{"accepted_marks": [<int>, ...]}</answer>`` block. Free-text
    reasoning may appear before the block but must not appear after.
    If multiple blocks appear we accept the last one (the MLLM's final
    answer). If ``valid_ids`` is provided, out-of-range ids are dropped.

    Returns an empty list if the response is missing the tag entirely or
    has a malformed payload -- callers should treat that as "no
    creatures accepted" rather than an error.

    The function is lenient on bad input: non-string ``text`` (including
    ``None``) returns an empty list rather than raising. Callers that have
    just gotten a model response that might be ``None`` due to upstream
    error get a safe default instead of an exception.
    """
    if not isinstance(text, str) or not text:
        return []
    matches = _ANSWER_RE.findall(text)
    if not matches:
        return []
    try:
        payload = json.loads(matches[-1])
    except json.JSONDecodeError:
        return []
    raw = payload.get("accepted_marks") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return []
    out: list[int] = []
    for item in raw:
        if isinstance(item, bool):
            continue
        if not isinstance(item, int):
            continue
        if valid_ids is not None and item not in valid_ids:
            continue
        out.append(item)
    return out
