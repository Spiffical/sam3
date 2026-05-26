# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import copy
import json
import os
import re

import cv2
from PIL import Image

from .client_llm import send_generate_request
from .client_sam3 import call_sam_service
from .helpers.mask_overlap_removal import remove_overlapping_masks
from .viz import visualize


def _read_prompt_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _load_system_prompts(current_dir):
    base_system_prompt_path = os.path.join(current_dir, "system_prompts/system_prompt.txt")
    base_iterative_prompt_path = os.path.join(
        current_dir, "system_prompts/system_prompt_iterative_checking.txt"
    )
    profile = os.environ.get("SAM3_AGENT_PROMPT_PROFILE", "general").strip().lower()
    system_prompt_override = os.environ.get("SAM3_SYSTEM_PROMPT_PATH", "").strip()
    iterative_prompt_override = os.environ.get(
        "SAM3_ITERATIVE_SYSTEM_PROMPT_PATH", ""
    ).strip()

    if system_prompt_override:
        system_prompt = _read_prompt_file(system_prompt_override)
    else:
        system_prompt = _read_prompt_file(base_system_prompt_path)

    if iterative_prompt_override:
        iterative_checking_system_prompt = _read_prompt_file(iterative_prompt_override)
    else:
        iterative_checking_system_prompt = _read_prompt_file(base_iterative_prompt_path)

    if profile == "underwater":
        underwater_system_addendum_path = os.path.join(
            current_dir, "system_prompts/system_prompt_underwater_addendum.txt"
        )
        underwater_iterative_addendum_path = os.path.join(
            current_dir,
            "system_prompts/system_prompt_iterative_checking_underwater_addendum.txt",
        )
        if not system_prompt_override and os.path.exists(underwater_system_addendum_path):
            system_prompt = (
                _read_prompt_file(underwater_system_addendum_path)
                + "\n\n"
                + system_prompt
            )
        if not iterative_prompt_override and os.path.exists(
            underwater_iterative_addendum_path
        ):
            iterative_checking_system_prompt = (
                _read_prompt_file(underwater_iterative_addendum_path)
                + "\n\n"
                + iterative_checking_system_prompt
            )
    elif profile not in {"", "general", "default", "base"}:
        print(
            f"[Warn] Unknown SAM3_AGENT_PROMPT_PROFILE='{profile}'. "
            "Falling back to base prompts."
        )

    return system_prompt, iterative_checking_system_prompt, profile


def save_debug_messages(messages_list, debug, debug_folder_path, debug_jsonl_path):
    """Save messages to debug jsonl file if debug is enabled"""
    if debug and debug_jsonl_path:
        # Ensure the debug directory exists before writing
        os.makedirs(debug_folder_path, exist_ok=True)
        with open(debug_jsonl_path, "w") as f:
            for msg in messages_list:
                f.write(json.dumps(msg, indent=4) + "\n")


def cleanup_debug_files(debug, debug_folder_path, debug_jsonl_path):
    """Clean up debug files when function successfully returns"""
    if debug and debug_folder_path:
        try:
            if os.path.exists(debug_jsonl_path):
                os.remove(debug_jsonl_path)
            if os.path.exists(debug_folder_path):
                os.rmdir(debug_folder_path)
        except Exception as e:
            print(f"Warning: Could not clean up debug files: {e}")


def count_images(messages):
    """Count the total number of images present in the messages history."""
    total = 0
    for message in messages:
        # Check if message has content (should be a list)
        if "content" in message and isinstance(message["content"], list):
            # Iterate through each content item
            for content_item in message["content"]:
                # Check if content item is a dict with type "image"
                if (
                    isinstance(content_item, dict)
                    and content_item.get("type") == "image"
                ):
                    total += 1
    return total


_ALLOWED_TOOL_NAMES = frozenset(
    {
        "segment_phrase",
        "drop_masks",
        "examine_each_mask",
        "select_masks_and_return",
        "report_no_mask",
    }
)


def _normalize_tool_call_dict(candidate):
    """Normalize candidate tool-call JSON into {'name': str, 'parameters': dict}."""
    if not isinstance(candidate, dict):
        return None

    tool_name = candidate.get("name") or candidate.get("tool_name")
    if not isinstance(tool_name, str) or tool_name not in _ALLOWED_TOOL_NAMES:
        return None

    parameters = candidate.get("parameters")
    if parameters is None and "arguments" in candidate:
        parameters = candidate.get("arguments")
    if isinstance(parameters, str):
        try:
            parameters = json.loads(parameters)
        except json.JSONDecodeError:
            return None
    if not isinstance(parameters, dict):
        return None

    return {"name": tool_name, "parameters": parameters}


def _extract_candidate_json_strings(generated_text):
    """Extract potential JSON objects from varied tool-call output formats."""
    candidates = []

    # Canonical format expected by our prompts.
    for tag in ("tool", "tool_call"):
        pattern = rf"<{tag}>(.*?)</{tag}>"
        for match in re.finditer(pattern, generated_text, flags=re.DOTALL):
            payload = match.group(1).strip()
            if payload:
                candidates.append(payload)

    # JSON fenced code blocks.
    for match in re.finditer(
        r"```(?:json)?\s*(\{.*?\})\s*```", generated_text, flags=re.DOTALL
    ):
        payload = match.group(1).strip()
        if payload:
            candidates.append(payload)

    # Any JSON object present in free text.
    decoder = json.JSONDecoder()
    for i, ch in enumerate(generated_text):
        if ch != "{":
            continue
        try:
            _, end_idx = decoder.raw_decode(generated_text[i:])
        except json.JSONDecodeError:
            continue
        payload = generated_text[i : i + end_idx].strip()
        if payload:
            candidates.append(payload)

    # De-duplicate while preserving order.
    return list(dict.fromkeys(candidates))


def _parse_tool_call_from_generated_text(generated_text):
    """Parse a tool call from generated text across multiple possible output shapes."""
    if not generated_text:
        return None

    for candidate_str in _extract_candidate_json_strings(generated_text):
        variations = [candidate_str]
        # Some model outputs include one extra trailing brace.
        if candidate_str.endswith("}}}"):
            variations.append(candidate_str[:-1])
        for item in variations:
            try:
                parsed = json.loads(item)
            except json.JSONDecodeError:
                continue
            normalized = _normalize_tool_call_dict(parsed)
            if normalized is None:
                continue
            canonical_text = f"<tool>{json.dumps(normalized, ensure_ascii=False)}</tool>"
            return normalized, canonical_text
    return None


def _last_user_has_image_context(messages):
    """Whether the most recent user message includes any image content item.

    Several follow-up tools (drop_masks, examine_each_mask) assume the prior
    user turn rendered an image of the current mask state. When the previous
    user turn is text-only (e.g. a format-repair retry or a "no new masks"
    response), invoking those tools used to index into a 1-element content
    list and raise IndexError. Callers should consult this helper before
    relying on that invariant.
    """
    if not messages:
        return False
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        return False
    content = last.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(item, dict) and item.get("type") == "image"
        for item in content
    )


def _build_invalid_tool_state_redirect_message(tool_name, initial_text_prompt):
    """Redirect message when a tool is invoked without a rendered-mask turn."""
    return (
        f"You called the {tool_name} tool, but there is no rendered mask "
        "image to act on in the most recent turn. Please call segment_phrase "
        "with a different, perhaps more general or more creative simple noun "
        "phrase text_prompt, or call report_no_mask if you do not believe any "
        f"targets exist. The original user query was: '{initial_text_prompt}'."
    )


def _redirect_for_invalid_tool_state(
    *, messages, generated_text, tool_name, initial_text_prompt
):
    """Append an assistant + user pair that redirects the model away from
    a tool invocation that cannot be served from the current context (e.g.
    drop_masks/examine_each_mask called after a text-only user turn).
    Mutates ``messages`` in place; the surrounding loop handles the next
    generation round.
    """
    print(
        f"⚠️ {tool_name} invoked without a rendered-mask image in the prior "
        "turn; redirecting the agent."
    )
    messages.append(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": generated_text}],
        }
    )
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": _build_invalid_tool_state_redirect_message(
                        tool_name, initial_text_prompt
                    ),
                }
            ],
        }
    )


def _build_tool_format_repair_message(path_to_latest_output_json, initial_text_prompt):
    if path_to_latest_output_json == "":
        valid_names = '["segment_phrase", "report_no_mask"]'
    else:
        valid_names = (
            '["segment_phrase", "drop_masks", "examine_each_mask", '
            '"select_masks_and_return", "report_no_mask"]'
        )
    return (
        "Your previous response did not contain a valid tool call JSON. "
        "Respond with exactly one tool call in this strict format and nothing else: "
        '<tool>{"name":"TOOL_NAME","parameters":{...}}</tool>. '
        f"The tool name must be one of: {valid_names}. "
        "Do not include analysis, markdown, bullet points, or any text outside <tool>...</tool>. "
        f"The original user query is: '{initial_text_prompt}'."
    )


def _is_broad_underwater_creature_query(text):
    if not isinstance(text, str):
        return False
    normalized = text.lower()
    keywords = (
        "small creature",
        "small creatures",
        "creature",
        "creatures",
        "marine organism",
        "marine organisms",
        "underwater",
    )
    return any(keyword in normalized for keyword in keywords)


def _safe_count_available_masks(path_to_latest_output_json):
    if not path_to_latest_output_json:
        return None
    try:
        with open(path_to_latest_output_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None

    pred_masks = payload.get("pred_masks")
    if isinstance(pred_masks, list):
        return len(pred_masks)
    pred_boxes = payload.get("pred_boxes")
    if isinstance(pred_boxes, list):
        return len(pred_boxes)
    return None


def _make_tool_call_text(tool_call):
    return f"<tool>{json.dumps(tool_call, ensure_ascii=False)}</tool>"


def _build_malformed_tool_call_fallback(
    *,
    path_to_latest_output_json,
    initial_text_prompt,
    prompt_profile,
    used_text_prompts,
):
    """
    Build a deterministic fallback tool call when the model repeatedly fails to emit
    valid tool-call JSON.

    Policy is controlled via SAM3_TOOL_CALL_FALLBACK_POLICY:
      - strict_fail (default for non-underwater): raise error
      - auto (default for underwater broad-creature prompts):
          * if masks exist => select all masks
          * if no masks => report_no_mask
          * first turn underwater broad-creature => segment_phrase with a safe phrase
      - select_all: force select all currently available masks when possible
      - report_no_mask: force report_no_mask
    """
    configured_policy = os.environ.get("SAM3_TOOL_CALL_FALLBACK_POLICY", "").strip().lower()
    if configured_policy:
        policy = configured_policy
    else:
        if (
            prompt_profile == "underwater"
            and _is_broad_underwater_creature_query(initial_text_prompt)
        ):
            policy = "auto"
        else:
            policy = "strict_fail"

    if policy in {"strict", "strict_fail", "off", "none"}:
        return None

    if policy in {"report", "report_no_mask"}:
        tool_call = {"name": "report_no_mask", "parameters": {}}
        return tool_call, _make_tool_call_text(tool_call)

    available_masks = _safe_count_available_masks(path_to_latest_output_json)
    if available_masks is not None and available_masks > 0:
        if policy in {"auto", "select_all", "auto_select_all"}:
            tool_call = {
                "name": "select_masks_and_return",
                "parameters": {"final_answer_masks": list(range(1, available_masks + 1))},
            }
            return tool_call, _make_tool_call_text(tool_call)

    if available_masks == 0 and policy in {"auto", "select_all", "auto_select_all"}:
        tool_call = {"name": "report_no_mask", "parameters": {}}
        return tool_call, _make_tool_call_text(tool_call)

    if not path_to_latest_output_json and policy in {"auto", "segment_phrase"}:
        for candidate_prompt in (
            "small creatures",
            "small creature",
            "creatures",
            "marine organisms",
            "small animals",
        ):
            if candidate_prompt not in used_text_prompts:
                tool_call = {
                    "name": "segment_phrase",
                    "parameters": {"text_prompt": candidate_prompt},
                }
                return tool_call, _make_tool_call_text(tool_call)
        # If all defaults were already used, finish gracefully.
        tool_call = {"name": "report_no_mask", "parameters": {}}
        return tool_call, _make_tool_call_text(tool_call)

    return None


def _terminate_for_cap_or_raise(
    *,
    path_to_latest_output_json,
    max_generations,
    initial_text_prompt,
    prompt_profile,
):
    """Gracefully end a frame when the generation cap is reached.

    Returns a terminating ``(tool_call, generated_text)`` tuple — either
    ``select_masks_and_return`` over all accumulated masks, or
    ``report_no_mask`` if none — so the caller can dispatch a final round
    without making another model call.

    Honours the same ``SAM3_TOOL_CALL_FALLBACK_POLICY`` env var as
    ``_build_malformed_tool_call_fallback``: when the resolved policy is
    ``strict_fail`` (the default for non-underwater prompts), this raises
    ``ValueError`` instead of terminating gracefully.
    """
    configured_policy = (
        os.environ.get("SAM3_TOOL_CALL_FALLBACK_POLICY", "").strip().lower()
    )
    if configured_policy:
        policy = configured_policy
    else:
        if (
            prompt_profile == "underwater"
            and _is_broad_underwater_creature_query(initial_text_prompt)
        ):
            policy = "auto"
        else:
            policy = "strict_fail"

    if policy in {"strict", "strict_fail", "off", "none"}:
        raise ValueError(
            f"Exceeded maximum number of allowed generation requests ({max_generations})"
        )

    available = _safe_count_available_masks(path_to_latest_output_json)
    if available and available > 0:
        tool_call = {
            "name": "select_masks_and_return",
            "parameters": {"final_answer_masks": list(range(1, available + 1))},
        }
    else:
        tool_call = {"name": "report_no_mask", "parameters": {}}
    return tool_call, _make_tool_call_text(tool_call)


def _extract_mask_verdict(generated_text):
    """Extract Accept/Reject verdict from mask-check text, if present."""
    if not isinstance(generated_text, str) or not generated_text.strip():
        return None

    tag_match = re.search(
        r"<\s*verdict\s*>\s*(accept|reject)\s*<\s*/\s*verdict\s*>",
        generated_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if tag_match:
        return tag_match.group(1).capitalize()

    # Fallback for non-tagged outputs: use the last explicit occurrence.
    plain_matches = re.findall(r"\b(accept|reject)\b", generated_text, flags=re.IGNORECASE)
    if plain_matches:
        return plain_matches[-1].capitalize()
    return None


def _compact_assistant_text_for_history(generated_text, max_chars=400):
    """
    Keep retry history compact to avoid context-window blowups.
    Preserve structured outputs when present; otherwise store a short summary.
    """
    if not isinstance(generated_text, str) or not generated_text.strip():
        return "[empty model response]"

    parsed_tool_call = _parse_tool_call_from_generated_text(generated_text)
    if parsed_tool_call is not None:
        return parsed_tool_call[1]

    verdict = _extract_mask_verdict(generated_text)
    if verdict is not None:
        return f"<verdict>{verdict}</verdict>"

    compact = generated_text.strip()
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars] + "\n...[truncated to reduce context size]..."


def _build_state_snapshot_paths(output_dir, img_path, tag):
    image_dir = os.path.join(output_dir, img_path.replace("/", "-"))
    os.makedirs(image_dir, exist_ok=True)
    safe_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", str(tag)).strip("._-") or "state"
    json_path = os.path.join(image_dir, f"{safe_tag}.json")
    image_path = os.path.join(image_dir, f"{safe_tag}.png")
    return json_path, image_path


def _persist_available_outputs(outputs, output_json_path):
    persisted = {
        "original_image_path": outputs["original_image_path"],
        "orig_img_h": outputs["orig_img_h"],
        "orig_img_w": outputs["orig_img_w"],
        "pred_boxes": list(outputs.get("pred_boxes", [])),
        "pred_scores": list(outputs.get("pred_scores", [])),
        "pred_masks": list(outputs.get("pred_masks", [])),
    }
    output_image_path = output_json_path.rsplit(".", 1)[0] + ".png"
    persisted["output_image_path"] = output_image_path
    rendered = visualize(persisted)
    rendered.save(output_image_path)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(persisted, f, indent=4)
    return persisted


def _merge_available_outputs(existing_outputs, new_outputs):
    if not existing_outputs or len(existing_outputs.get("pred_masks", [])) == 0:
        merged = {
            "original_image_path": new_outputs["original_image_path"],
            "orig_img_h": new_outputs["orig_img_h"],
            "orig_img_w": new_outputs["orig_img_w"],
            "pred_boxes": list(new_outputs.get("pred_boxes", [])),
            "pred_scores": list(new_outputs.get("pred_scores", [])),
            "pred_masks": list(new_outputs.get("pred_masks", [])),
        }
        return remove_overlapping_masks(merged)

    if len(new_outputs.get("pred_masks", [])) == 0:
        return {
            "original_image_path": existing_outputs["original_image_path"],
            "orig_img_h": existing_outputs["orig_img_h"],
            "orig_img_w": existing_outputs["orig_img_w"],
            "pred_boxes": list(existing_outputs.get("pred_boxes", [])),
            "pred_scores": list(existing_outputs.get("pred_scores", [])),
            "pred_masks": list(existing_outputs.get("pred_masks", [])),
        }

    merged = {
        "original_image_path": existing_outputs["original_image_path"],
        "orig_img_h": existing_outputs["orig_img_h"],
        "orig_img_w": existing_outputs["orig_img_w"],
        "pred_boxes": list(existing_outputs.get("pred_boxes", []))
        + list(new_outputs.get("pred_boxes", [])),
        "pred_scores": list(existing_outputs.get("pred_scores", []))
        + list(new_outputs.get("pred_scores", [])),
        "pred_masks": list(existing_outputs.get("pred_masks", []))
        + list(new_outputs.get("pred_masks", [])),
    }
    merged = remove_overlapping_masks(merged)
    return {
        "original_image_path": merged["original_image_path"],
        "orig_img_h": merged["orig_img_h"],
        "orig_img_w": merged["orig_img_w"],
        "pred_boxes": list(merged.get("pred_boxes", [])),
        "pred_scores": list(merged.get("pred_scores", [])),
        "pred_masks": list(merged.get("pred_masks", [])),
    }


def _request_mask_verdict_with_retry(send_generate_request_fn, iterative_messages, max_retries=2):
    """
    Request a mask verdict and retry with strict formatting instructions if missing.

    Returns:
        tuple[str | None, str | None]: (model_text, verdict)
    """
    model_text = send_generate_request_fn(iterative_messages)
    verdict = _extract_mask_verdict(model_text)
    if verdict is not None:
        return model_text, verdict

    for retry_idx in range(max_retries):
        iterative_messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": _compact_assistant_text_for_history(model_text),
                    }
                ],
            }
        )
        iterative_messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Your previous response did not include a valid verdict. "
                            "Respond with exactly one of these tags and nothing else: "
                            "<verdict>Accept</verdict> or <verdict>Reject</verdict>."
                        ),
                    }
                ],
            }
        )
        model_text = send_generate_request_fn(iterative_messages)
        verdict = _extract_mask_verdict(model_text)
        if verdict is not None:
            return model_text, verdict
        print(
            "⚠️ Missing mask verdict format. "
            f"Retrying iterative-check verdict extraction ({retry_idx + 1}/{max_retries})."
        )

    return model_text, None


def _prune_messages_for_next_round(
    messages_list,
    used_text_prompts,
    latest_sam3_text_prompt,
    img_path,
    initial_text_prompt,
):
    """Return a new messages list that contains only:
    1) messages[:2] (with optional warning text added to the second message's content)
    2) the latest assistant message (and everything after it) that contains a segment_phrase tool call
    """
    # Allow some slack for temporary format-repair retries before pruning.
    assert len(messages_list) <= 20

    # Part 1: always keep the first two message JSONs
    part1 = copy.deepcopy(messages_list[:2])

    # Part 2: search backwards for the latest assistant message containing a segment_phrase tool call
    part2_start_idx = None
    for idx in range(len(messages_list) - 1, 1, -1):
        msg = messages_list[idx]
        # We only consider assistant messages with a "content" list
        if msg.get("role") != "assistant" or "content" not in msg:
            continue
        # Look for any content element that is a text containing the segment_phrase tool call
        for content in msg["content"]:
            if (
                isinstance(content, dict)
                and content.get("type") == "text"
                and "<tool>" in content.get("text", "")
                and "segment_phrase" in content.get("text", "")
            ):
                part2_start_idx = idx
                break
        if part2_start_idx is not None:
            break

    part2 = messages_list[part2_start_idx:] if part2_start_idx is not None else []

    # Part 3: decide whether to add warning text to the second message in part1
    previously_used = (
        [p for p in used_text_prompts if p != latest_sam3_text_prompt]
        if latest_sam3_text_prompt
        else list(used_text_prompts)
    )
    if part2 and len(previously_used) > 0:
        warning_text = f'Note that we have previously called the segment_phrase tool with each "text_prompt" in this list: {list(previously_used)}. Do not use any of these phrases again as the "text_prompt" for segment_phrase.'
        # Replace the second message entirely to keep exactly 2 content items
        part1[1] = {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {
                    "type": "text",
                    "text": f"The above image is the raw input image. The initial user input query is: '{initial_text_prompt}'."
                    + " "
                    + warning_text,
                },
            ],
        }
        assert len(part1[1]["content"]) == 2

    # Build the new messages list: part1 (with optional warning), then part2
    new_messages = list(part1)
    new_messages.extend(part2)
    return new_messages


def agent_inference(
    img_path: str,
    initial_text_prompt: str,
    debug: bool = False,
    send_generate_request=send_generate_request,
    call_sam_service=call_sam_service,
    max_generations: int = 100,
    output_dir="../../sam3_agent_out",
):
    """
    Given a text prompt and an image, this tool will perform all aspects of agentic problem solving,
    while saving sam3 and MLLM outputs to their respective directories.

    Args:
        img_path: Path to the input image
        initial_text_prompt: Initial text prompt from the user
        debug: Whether to enable debug mode
        max_generations: Maximum number of send_generate_request calls allowed (default: 100)
    """
    # setup dir
    sam_output_dir = os.path.join(output_dir, "sam_out")
    error_save_dir = os.path.join(output_dir, "none_out")
    debug_save_dir = os.path.join(output_dir, "agent_debug_out")
    os.makedirs(sam_output_dir, exist_ok=True)
    os.makedirs(error_save_dir, exist_ok=True)
    os.makedirs(debug_save_dir, exist_ok=True)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # init variables
    PATH_TO_LATEST_OUTPUT_JSON = ""
    LATEST_SAM3_TEXT_PROMPT = ""
    USED_TEXT_PROMPTS = (
        set()
    )  # Track all previously used text prompts for segment_phrase
    generation_count = 0  # Counter for number of send_generate_request calls

    # debug setup
    debug_folder_path = None
    debug_jsonl_path = None
    if debug:
        debug_folder_path = os.path.join(
            debug_save_dir, f"{img_path.rsplit('/', 1)[-1].rsplit('.', 1)[0]}"
        )
        debug_jsonl_path = os.path.join(debug_folder_path, "debug_history.json")
        os.makedirs(debug_folder_path, exist_ok=True)

    # The helper functions are now defined outside the agent_inference function
    (
        system_prompt,
        iterative_checking_system_prompt,
        prompt_profile,
    ) = _load_system_prompts(current_dir)
    print(f"> Prompt profile: {prompt_profile}")

    # Construct the initial message list
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {
                    "type": "text",
                    "text": f"The above image is the raw input image. The initial user input query is: '{initial_text_prompt}'.",
                },
            ],
        },
    ]
    print(f"> Text prompt: {initial_text_prompt}")
    print(f"> Image path: {img_path}")

    print("\n\n")
    print("-" * 30 + f" Round {str(generation_count + 1)}" + "-" * 30)
    print("\n\n")
    generated_text = send_generate_request(messages)
    print(f"\n>>> MLLM Response [start]\n{generated_text}\n<<< MLLM Response [end]\n")
    malformed_tool_call_retries = 0
    while generated_text is not None:
        save_debug_messages(messages, debug, debug_folder_path, debug_jsonl_path)
        parsed_tool_call = _parse_tool_call_from_generated_text(generated_text)
        if parsed_tool_call is None:
            malformed_tool_call_retries += 1
            if malformed_tool_call_retries > 3:
                fallback_tool = _build_malformed_tool_call_fallback(
                    path_to_latest_output_json=PATH_TO_LATEST_OUTPUT_JSON,
                    initial_text_prompt=initial_text_prompt,
                    prompt_profile=prompt_profile,
                    used_text_prompts=USED_TEXT_PROMPTS,
                )
                if fallback_tool is None:
                    raise ValueError(f"Invalid JSON in tool call: {generated_text}")

                tool_call, generated_text = fallback_tool
                malformed_tool_call_retries = 0
                print(
                    "⚠️ Invalid tool-call format persisted after retries. "
                    f"Applying fallback tool call: {tool_call}"
                )
            else:
                print(
                    "⚠️ Invalid tool-call format from model. "
                    f"Requesting a strict-format retry ({malformed_tool_call_retries}/3)."
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": _compact_assistant_text_for_history(generated_text),
                            }
                        ],
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": _build_tool_format_repair_message(
                                    PATH_TO_LATEST_OUTPUT_JSON, initial_text_prompt
                                ),
                            }
                        ],
                    }
                )

                # Keep parse retries bounded within max_generations as well.
                generation_count += 1
                if generation_count > max_generations:
                    _cap_tool_call, generated_text = _terminate_for_cap_or_raise(
                        path_to_latest_output_json=PATH_TO_LATEST_OUTPUT_JSON,
                        max_generations=max_generations,
                        initial_text_prompt=initial_text_prompt,
                        prompt_profile=prompt_profile,
                    )
                    print(
                        "⚠️ Max generations exceeded during format-repair retries. "
                        f"Forcing terminating fallback: {_cap_tool_call}"
                    )
                else:
                    print("\n\n")
                    print("-" * 30 + f" Round {str(generation_count + 1)}" + "-" * 30)
                    print("\n\n")
                    generated_text = send_generate_request(messages)
                    print(
                        f"\n>>> MLLM Response [start]\n{generated_text}\n<<< MLLM Response [end]\n"
                    )
                continue

        else:
            malformed_tool_call_retries = 0
            tool_call, generated_text = parsed_tool_call

        if PATH_TO_LATEST_OUTPUT_JSON == "":
            # The first tool call must be segment_phrase or report_no_mask
            assert (
                tool_call["name"] == "segment_phrase"
                or tool_call["name"] == "report_no_mask"
            )

        if tool_call["name"] == "segment_phrase":
            print("🔍 Calling segment_phrase tool...")
            assert list(tool_call["parameters"].keys()) == ["text_prompt"]

            # Check if this text_prompt has been used before
            current_text_prompt = tool_call["parameters"]["text_prompt"]
            if current_text_prompt in USED_TEXT_PROMPTS:
                print(
                    f"❌ Text prompt '{current_text_prompt}' has been used before. Requesting a different prompt."
                )
                duplicate_prompt_message = f"You have previously used '{current_text_prompt}' as your text_prompt to call the segment_phrase tool. You may not use it again. Please call the segment_phrase tool again with a different, perhaps more general, or more creative simple noun phrase prompt, while adhering to all the rules stated in the system prompt. You must also never use any of the following text_prompt(s): {str(list(USED_TEXT_PROMPTS))}."
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": generated_text}],
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": duplicate_prompt_message}],
                    }
                )
            else:
                # Add the text_prompt to the set of used prompts
                USED_TEXT_PROMPTS.add(current_text_prompt)
                LATEST_SAM3_TEXT_PROMPT = current_text_prompt
                latest_phrase_output_json = call_sam_service(
                    image_path=img_path,
                    text_prompt=current_text_prompt,
                    output_folder_path=sam_output_dir,
                )
                sam3_outputs = json.load(open(latest_phrase_output_json, "r"))
                num_new_masks = len(sam3_outputs["pred_boxes"])
                existing_outputs = (
                    json.load(open(PATH_TO_LATEST_OUTPUT_JSON, "r"))
                    if PATH_TO_LATEST_OUTPUT_JSON
                    else None
                )
                merged_outputs = _merge_available_outputs(existing_outputs, sam3_outputs)
                total_masks = len(merged_outputs["pred_boxes"])
                state_json_path, _ = _build_state_snapshot_paths(
                    sam_output_dir,
                    img_path,
                    f"available_masks_round_{generation_count + 1}",
                )
                merged_outputs = _persist_available_outputs(merged_outputs, state_json_path)
                PATH_TO_LATEST_OUTPUT_JSON = state_json_path
                sam3_output_image_path = merged_outputs["output_image_path"]

                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": generated_text}],
                    }
                )
                if num_new_masks == 0 and total_masks == 0:
                    print("❌ No masks generated by SAM3, reporting no mask to Qwen.")
                    sam3_output_text_message = f"The segment_phrase tool did not generate any masks for the text_prompt '{current_text_prompt}', and there are still no available masks in memory. Now, please call the segment_phrase tool again with a different, perhaps more general, or more creative simple noun phrase text_prompt, while adhering to all the rules stated in the system prompt. Please be reminded that the original user query was '{initial_text_prompt}'."
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": sam3_output_text_message}
                            ],
                        }
                    )
                elif num_new_masks == 0:
                    sam3_output_text_message = f"The segment_phrase tool did not generate any new masks for the text_prompt '{current_text_prompt}'. However, your current memory still contains {total_masks} available mask(s) accumulated from previous turns. All {total_masks} currently available mask(s) are rendered in the image below. You may keep them, delete some of them, or continue searching with another segment_phrase call. Please be reminded that the original user query was '{initial_text_prompt}'."
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": sam3_output_text_message},
                                {"type": "image", "image": sam3_output_image_path},
                            ],
                        }
                    )
                else:
                    sam3_output_text_message = rf"The segment_phrase tool generated {num_new_masks} new mask(s) for the text_prompt '{current_text_prompt}'. Your current memory now contains {total_masks} available mask(s) accumulated across turns. All {total_masks} currently available mask(s) are rendered in this image below. Now you must analyze the available mask(s) carefully, compare them against the raw input image and the original user query, and determine your next action. You may keep accumulating more masks with segment_phrase, delete incorrect masks with drop_masks, inspect masks with examine_each_mask, or finish with select_masks_and_return. Please be reminded that the original user query was '{initial_text_prompt}'."
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": sam3_output_text_message},
                                {"type": "image", "image": sam3_output_image_path},
                            ],
                        }
                    )
                print("\n\n>>> sam3_output_text_message:\n", sam3_output_text_message)

        elif tool_call["name"] == "drop_masks":
            print("🔍 Calling drop_masks tool...")
            assert PATH_TO_LATEST_OUTPUT_JSON != ""
            assert list(tool_call["parameters"].keys()) == ["mask_indices_to_drop"]
            if not _last_user_has_image_context(messages):
                _redirect_for_invalid_tool_state(
                    messages=messages,
                    generated_text=generated_text,
                    tool_name="drop_masks",
                    initial_text_prompt=initial_text_prompt,
                )
            else:
                messages.pop()
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "There are currently available masks in memory. You are deleting some of them and then must re-evaluate the remaining masks.",
                            }
                        ],
                    }
                )
                current_outputs = json.load(open(PATH_TO_LATEST_OUTPUT_JSON, "r"))
                requested_drops = tool_call["parameters"]["mask_indices_to_drop"]
                available_masks = set(range(1, len(current_outputs["pred_masks"]) + 1))
                masks_to_drop = sorted({i for i in requested_drops if i in available_masks})
                masks_to_keep = sorted(i for i in available_masks if i not in set(masks_to_drop))

                updated_outputs = {
                    "original_image_path": current_outputs["original_image_path"],
                    "orig_img_h": current_outputs["orig_img_h"],
                    "orig_img_w": current_outputs["orig_img_w"],
                    "pred_boxes": [current_outputs["pred_boxes"][i - 1] for i in masks_to_keep],
                    "pred_scores": [current_outputs["pred_scores"][i - 1] for i in masks_to_keep],
                    "pred_masks": [current_outputs["pred_masks"][i - 1] for i in masks_to_keep],
                }
                state_json_path, _ = _build_state_snapshot_paths(
                    sam_output_dir,
                    img_path,
                    f"available_masks_after_drop_round_{generation_count + 1}",
                )
                updated_outputs = _persist_available_outputs(updated_outputs, state_json_path)
                PATH_TO_LATEST_OUTPUT_JSON = state_json_path

                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": generated_text}],
                    }
                )
                if len(masks_to_keep) == 0:
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"The original user query was: '{initial_text_prompt}'. The drop_masks tool removed all currently available masks from memory. There are now 0 available masks. If you still believe target objects exist, call segment_phrase again with a new text_prompt. Otherwise, you may call report_no_mask.",
                                }
                            ],
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"The original user query was: '{initial_text_prompt}'. The drop_masks tool removed mask(s) {masks_to_drop}. There are now {len(masks_to_keep)} available mask(s) left in memory. All remaining available mask(s) are rendered in this image below. Analyze them carefully and determine your next action.",
                                },
                                {"type": "image", "image": updated_outputs["output_image_path"]},
                            ],
                        }
                    )

        elif tool_call["name"] == "examine_each_mask":
            print("🔍 Calling examine_each_mask tool...")
            assert LATEST_SAM3_TEXT_PROMPT != ""

            if not _last_user_has_image_context(messages):
                _redirect_for_invalid_tool_state(
                    messages=messages,
                    generated_text=generated_text,
                    tool_name="examine_each_mask",
                    initial_text_prompt=initial_text_prompt,
                )
            else:
                messages.pop()  # Remove the last user message
                # Add simplified replacement message
                simplified_message = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "There are several currently available masks in memory. Now you must analyze the mask(s) carefully, compare them against the raw input image and the original user query, and determine your next action.",
                        }
                    ],
                }
                messages.append(simplified_message)

                current_outputs = json.load(open(PATH_TO_LATEST_OUTPUT_JSON, "r"))
                num_masks = len(current_outputs["pred_masks"])
                masks_to_keep = []

                # MLLM check the mask one by one
                for i in range(num_masks):
                    print(f"🔍 Checking mask {i + 1}/{num_masks}...")
                    image_w_mask_i, image_w_zoomed_in_mask_i = visualize(current_outputs, i)

                    image_w_zoomed_in_mask_i_path = os.path.join(
                        sam_output_dir, rf"{LATEST_SAM3_TEXT_PROMPT}.png".replace("/", "_")
                    ).replace(".png", f"_zoom_in_mask_{i + 1}.png")
                    image_w_mask_i_path = os.path.join(
                        sam_output_dir, rf"{LATEST_SAM3_TEXT_PROMPT}.png".replace("/", "_")
                    ).replace(".png", f"_selected_mask_{i + 1}.png")
                    image_w_zoomed_in_mask_i.save(image_w_zoomed_in_mask_i_path)
                    image_w_mask_i.save(image_w_mask_i_path)

                    iterative_checking_messages = [
                        {"role": "system", "content": iterative_checking_system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"The raw input image: "},
                                {"type": "image", "image": img_path},
                                {
                                    "type": "text",
                                    "text": f"The initial user input query is: '{initial_text_prompt}'",
                                },
                                {
                                    "type": "text",
                                    "text": f"Image with the predicted segmentation mask rendered on it: ",
                                },
                                {"type": "image", "image": image_w_mask_i_path},
                                {
                                    "type": "text",
                                    "text": f"Image with the zoomed-in mask: ",
                                },
                                {"type": "image", "image": image_w_zoomed_in_mask_i_path},
                            ],
                        },
                    ]
                    checking_generated_text, verdict = _request_mask_verdict_with_retry(
                        send_generate_request,
                        iterative_checking_messages,
                        max_retries=2,
                    )

                    # Process the generated text to determine if the mask should be kept or rejected
                    if checking_generated_text is None:
                        raise ValueError(
                            "Generated text is None, which is unexpected. Please check the Qwen server and the input parameters."
                        )
                    print(f"Generated text for mask {i + 1}: {checking_generated_text}")
                    if verdict is None:
                        fallback_verdict = (
                            os.environ.get("SAM3_MASK_CHECK_DEFAULT_VERDICT", "Reject")
                            .strip()
                            .capitalize()
                        )
                        if fallback_verdict not in {"Accept", "Reject"}:
                            fallback_verdict = "Reject"
                        print(
                            "⚠️ Could not parse Accept/Reject verdict after retries. "
                            f"Falling back to {fallback_verdict} for mask {i + 1}."
                        )
                        verdict = fallback_verdict

                    if verdict == "Accept":
                        print(f"Mask {i + 1} accepted, keeping it in the outputs.")
                        masks_to_keep.append(i)
                    elif verdict == "Reject":
                        print(f"Mask {i + 1} rejected, removing it from the outputs.")
                    else:
                        raise ValueError(
                            f"Unexpected verdict value '{verdict}' for generated text: {checking_generated_text}. Expected 'Accept' or 'Reject'."
                        )

                updated_outputs = {
                    "original_image_path": current_outputs["original_image_path"],
                    "orig_img_h": current_outputs["orig_img_h"],
                    "orig_img_w": current_outputs["orig_img_w"],
                    "pred_boxes": [current_outputs["pred_boxes"][i] for i in masks_to_keep],
                    "pred_scores": [
                        current_outputs["pred_scores"][i] for i in masks_to_keep
                    ],
                    "pred_masks": [current_outputs["pred_masks"][i] for i in masks_to_keep],
                }
                state_json_path, _ = _build_state_snapshot_paths(
                    sam_output_dir,
                    img_path,
                    f"available_masks_after_examine_round_{generation_count + 1}",
                )
                updated_outputs = _persist_available_outputs(updated_outputs, state_json_path)
                # save the updated json outputs and append to message history
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": generated_text}],
                    }
                )
                if len(masks_to_keep) == 0:
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"The original user query was: '{initial_text_prompt}'. The examine_each_mask tool examined and rejected all currently available masks in memory. There are now 0 available masks. If you still believe target objects exist, please call the segment_phrase tool again with a different, perhaps more general, or more creative simple noun phrase text_prompt, while adhering to all the rules stated in the system prompt.",
                                }
                            ],
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"The original user query was: '{initial_text_prompt}'. After calling the examine_each_mask tool on the available masks, the number of available masks is now {len(masks_to_keep)}. All {len(masks_to_keep)} available masks are rendered in this image below, now you must analyze the {len(masks_to_keep)} available mask(s) carefully, compare them against the raw input image and the original user query, and determine your next action.",
                                },
                                {"type": "image", "image": updated_outputs["output_image_path"]},
                            ],
                        }
                    )

                PATH_TO_LATEST_OUTPUT_JSON = state_json_path

        elif tool_call["name"] == "select_masks_and_return":
            print("🔍 Calling select_masks_and_return tool...")
            current_outputs = json.load(open(PATH_TO_LATEST_OUTPUT_JSON, "r"))

            assert list(tool_call["parameters"].keys()) == ["final_answer_masks"]
            masks_to_keep = tool_call["parameters"]["final_answer_masks"]

            # Keep only valid mask indices, remove duplicates, and preserve deterministic ascending order
            available_masks = set(range(1, len(current_outputs["pred_masks"]) + 1))
            masks_to_keep = sorted({i for i in masks_to_keep if i in available_masks})
            # Change this to a update message telling the model to try again along with information about errors made.

            final_outputs = {
                "original_image_path": current_outputs["original_image_path"],
                "orig_img_h": current_outputs["orig_img_h"],
                "orig_img_w": current_outputs["orig_img_w"],
                "pred_boxes": [
                    current_outputs["pred_boxes"][i - 1] for i in masks_to_keep
                ],
                "pred_scores": [
                    current_outputs["pred_scores"][i - 1] for i in masks_to_keep
                ],
                "pred_masks": [
                    current_outputs["pred_masks"][i - 1] for i in masks_to_keep
                ],
            }

            rendered_final_output = visualize(final_outputs)
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": generated_text}],
                }
            )

            # NOTE: debug files (when debug=True) are intentionally retained
            # so scripts/analyze_agent_run.py can parse the per-frame tool
            # sequence after a successful run.
            return messages, final_outputs, rendered_final_output

        elif tool_call["name"] == "report_no_mask":
            print("🔍 Calling report_no_mask tool...")
            height, width = cv2.imread(img_path).shape[:2]
            final_outputs = {
                "original_image_path": img_path,
                "orig_img_h": height,
                "orig_img_w": width,
                "pred_boxes": [],
                "pred_scores": [],
                "pred_masks": [],
            }
            rendered_final_output = Image.open(img_path)
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": generated_text}],
                }
            )
            return messages, final_outputs, rendered_final_output

        else:
            raise ValueError(f"Unknown tool call: {tool_call['name']}")

        # sometimes the MLLM don't know when to stop, and generates multiple tool calls in one round, so we need to split the generated text by </tool> and only keep the first one

        for message in messages:
            if message["role"] == "assistant" and "content" in message:
                for content in message["content"]:
                    if (
                        isinstance(content, dict)
                        and content.get("type") == "text"
                        and "text" in content
                    ):
                        content["text"] = (
                            content["text"].split("</tool>", 1)[0] + "</tool>\n\n"
                        )
        # Prune the messages history before the next MLLM generation round according to the 3-part rules.
        # This keeps history compact and ensures the model sees only the allowed parts.
        messages = _prune_messages_for_next_round(
            messages,
            USED_TEXT_PROMPTS,
            LATEST_SAM3_TEXT_PROMPT,
            img_path,
            initial_text_prompt,
        )
        # make sure there can never be more than 2 images in the context
        assert count_images(messages) <= 2
        generation_count += 1
        if generation_count > max_generations:
            _cap_tool_call, generated_text = _terminate_for_cap_or_raise(
                path_to_latest_output_json=PATH_TO_LATEST_OUTPUT_JSON,
                max_generations=max_generations,
                initial_text_prompt=initial_text_prompt,
                prompt_profile=prompt_profile,
            )
            print(
                "⚠️ Max generations exceeded. "
                f"Forcing terminating fallback: {_cap_tool_call}"
            )
        else:
            print("\n\n")
            print("-" * 30 + f" Round {str(generation_count + 1)}" + "-" * 30)
            print("\n\n")
            generated_text = send_generate_request(messages)
            print(
                f"\n>>> MLLM Response [start]\n{generated_text}\n<<< MLLM Response [end]\n"
            )

    print("\n\n>>> SAM 3 Agent execution ended.\n\n")

    error_save_path = os.path.join(
        error_save_dir,
        f"{img_path.rsplit('/', 1)[-1].rsplit('.', 1)[0]}_error_history.json",
    )
    with open(error_save_path, "w") as f:
        json.dump(messages, f, indent=4)
    print("Saved messages history that caused error to:", error_save_path)
    raise ValueError(
        rf"Generated text is None, which is unexpected. Please check the Qwen server and the input parameters for image path: {img_path} and initial text prompt: {initial_text_prompt}."
    )
