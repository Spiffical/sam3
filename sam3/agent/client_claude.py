# pyre-unsafe

"""Anthropic Claude adapter for the SAM3 MLLM agent.

Exposes `send_claude_request(messages, ...)` with the same calling
convention the agent loop uses for the OpenAI-compatible path in
`client_llm.send_generate_request`, so the existing agent_core parser
that looks for `<tool>{...}</tool>` text continues to work unchanged.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

import anthropic

from .client_llm import (
    _cap_images_in_processed_messages,
    get_image_base64_and_mime,
)


def _flatten_system_content(content: Any) -> str:
    """The agent stores system content as a plain string today, but be lenient."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text", "")))
            elif isinstance(item, str):
                chunks.append(item)
        return "\n".join(chunks)
    return str(content or "")


def _normalize_user_or_assistant_content(content: Any) -> Any:
    """
    Pass through Anthropic-style content lists; convert OpenAI-style
    string-or-list content into list-of-blocks the converter expects.
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return content
    if content is None:
        return []
    return [{"type": "text", "text": str(content)}]


def _convert_content_block_to_anthropic(
    block: Any,
    *,
    image_max_edge: Optional[int],
) -> Optional[dict[str, Any]]:
    """
    Convert one OpenAI-style content block into an Anthropic content block.

    Inputs we might see from agent_core.py:
      {"type": "text", "text": "..."}
      {"type": "image", "image": "/path/to/frame.jpg"}
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,...", "detail": "..."}}
    """
    if not isinstance(block, dict):
        if isinstance(block, str) and block.strip():
            return {"type": "text", "text": block}
        return None

    btype = str(block.get("type", "")).lower()

    if btype in {"text", "output_text"}:
        text = block.get("text")
        if not isinstance(text, str) or not text.strip():
            return None
        return {"type": "text", "text": text}

    if btype == "image":
        image_path = block.get("image")
        if not isinstance(image_path, str) or not image_path:
            return None
        safe_path = image_path.replace("?", "%3F")
        b64, mime = get_image_base64_and_mime(safe_path, max_edge=image_max_edge)
        if not b64:
            return None
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": mime, "data": b64},
        }

    if btype == "image_url":
        url_field = block.get("image_url")
        url = url_field.get("url") if isinstance(url_field, dict) else url_field
        if not isinstance(url, str):
            return None
        if url.startswith("data:"):
            try:
                header, payload = url.split(",", 1)
                mime = header.split(";")[0][len("data:"):] or "image/jpeg"
            except ValueError:
                return None
            return {
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": payload},
            }
        # URL-based images aren't expected from this codepath.
        return None

    return None


def _convert_messages_to_anthropic(
    messages: list[dict[str, Any]],
    *,
    image_max_edge: Optional[int],
    max_images: Optional[int],
) -> tuple[str, list[dict[str, Any]]]:
    """
    Split system messages out, optionally cap the number of images carried in
    user messages, and convert each user/assistant message into the
    Anthropic Messages API content-block shape.
    """
    # First pass: lift system messages, normalize user/assistant content to lists.
    system_chunks: list[str] = []
    normalized: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            piece = _flatten_system_content(msg.get("content"))
            if piece.strip():
                system_chunks.append(piece)
            continue
        if role not in {"user", "assistant"}:
            continue
        normalized.append(
            {
                "role": role,
                "content": _normalize_user_or_assistant_content(msg.get("content")),
            }
        )

    # Image cap is computed against the OpenAI-style content (before base64 inflation).
    capped = _cap_images_in_processed_messages(
        [
            # _cap_images_in_processed_messages keys off content[i]["type"] == "image_url",
            # so we temporarily relabel "image" blocks the same way before capping
            # then restore them. This avoids reimplementing the position bookkeeping.
            {
                "role": m["role"],
                "content": [
                    {"type": "image_url", "_orig": part}
                    if isinstance(part, dict) and part.get("type") == "image"
                    else part
                    for part in m["content"]
                ],
            }
            for m in normalized
        ],
        max_images,
    )
    restored: list[dict[str, Any]] = []
    for m in capped:
        new_parts = []
        for part in m["content"]:
            if isinstance(part, dict) and part.get("type") == "image_url" and "_orig" in part:
                new_parts.append(part["_orig"])
            else:
                new_parts.append(part)
        restored.append({"role": m["role"], "content": new_parts})

    # Second pass: convert each content block to Anthropic shape.
    anthropic_messages: list[dict[str, Any]] = []
    for msg in restored:
        converted_blocks: list[dict[str, Any]] = []
        for block in msg["content"]:
            converted = _convert_content_block_to_anthropic(
                block, image_max_edge=image_max_edge
            )
            if converted is not None:
                converted_blocks.append(converted)
        if not converted_blocks:
            continue
        anthropic_messages.append({"role": msg["role"], "content": converted_blocks})

    system_text = "\n\n".join(system_chunks).strip()
    return system_text, anthropic_messages


def _extract_text_from_response(response: Any) -> Optional[str]:
    blocks = getattr(response, "content", None)
    if not blocks:
        return None
    out: list[str] = []
    for blk in blocks:
        btype = getattr(blk, "type", None)
        if btype == "text":
            text = getattr(blk, "text", "")
            if isinstance(text, str) and text.strip():
                out.append(text)
        elif isinstance(blk, dict):
            if blk.get("type") == "text":
                text = blk.get("text", "")
                if isinstance(text, str) and text.strip():
                    out.append(text)
    if not out:
        return None
    return "\n".join(out).strip()


def _next_smaller_image_edge(
    current_edge: Optional[int], floor_edge: int
) -> Optional[int]:
    if current_edge is None or current_edge <= floor_edge:
        return None
    reduced = max(floor_edge, int(current_edge * 0.8))
    if reduced >= current_edge:
        return None
    return reduced


def send_claude_request(
    messages: list[dict[str, Any]],
    *,
    model: str = "claude-sonnet-4-6",
    api_key: Optional[str] = None,
    max_tokens: int = 1024,
    effort: Optional[str] = None,
    server_url: Optional[str] = None,  # accepted for signature symmetry; ignored
    enable_prompt_cache: bool = True,
    max_retries: int = 5,
) -> Optional[str]:
    """
    Send a chat-style message list to Claude and return the assistant text.

    Args mirror `client_llm.send_generate_request` so this function can be
    bound with `functools.partial` and passed in as the `send_generate_request`
    argument of `agent_inference` without further plumbing.

    `server_url` is accepted and ignored to keep the call site symmetric with
    the OpenAI-compatible adapter.
    """
    del server_url  # unused

    resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not resolved_key:
        print("Anthropic request failed: ANTHROPIC_API_KEY is not set.")
        return None
    resolved_effort = (effort or os.environ.get("ANTHROPIC_EFFORT", "")).strip().lower()
    allowed_efforts = {"low", "medium", "high", "xhigh", "max"}
    if resolved_effort and resolved_effort not in allowed_efforts:
        print(
            "Anthropic request failed: effort must be one of "
            f"{sorted(allowed_efforts)}, got {resolved_effort!r}."
        )
        return None
    try:
        timeout_seconds = float(os.environ.get("ANTHROPIC_TIMEOUT_SECONDS", "120"))
    except ValueError:
        timeout_seconds = 120.0
    client = anthropic.Anthropic(
        api_key=resolved_key,
        timeout=max(10.0, timeout_seconds),
    )

    try:
        image_max_edge = int(os.environ.get("SAM3_AGENT_IMAGE_MAX_EDGE", "1024"))
    except ValueError:
        image_max_edge = 1024
    try:
        image_min_edge = int(os.environ.get("SAM3_AGENT_IMAGE_MIN_EDGE", "384"))
    except ValueError:
        image_min_edge = 384
    image_min_edge = max(128, image_min_edge)

    forced_max_images: Optional[int] = None
    max_images_env = os.environ.get("SAM3_MAX_IMAGES_PER_REQUEST")
    if max_images_env:
        try:
            forced_max_images = int(max_images_env)
        except ValueError:
            forced_max_images = None

    current_edge: Optional[int] = image_max_edge if image_max_edge > 0 else None
    budget = max(int(max_tokens), 1)

    for attempt in range(max_retries):
        system_text, anthropic_messages = _convert_messages_to_anthropic(
            messages,
            image_max_edge=current_edge,
            max_images=forced_max_images,
        )

        if not anthropic_messages:
            print("Anthropic request skipped: no convertible user/assistant messages.")
            return None

        # Attach an ephemeral prompt-cache marker to the last block of the
        # system prompt so the agent's stable, multi-turn preamble doesn't
        # re-tokenize on every frame. Anthropic caches the prefix up to and
        # including the marked block.
        system_payload: Any = system_text
        if enable_prompt_cache and system_text:
            system_payload = [
                {
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        effort_suffix = f", effort={resolved_effort}" if resolved_effort else ""
        print(
            f"[Claude] Calling model {model} "
            f"(attempt {attempt + 1}/{max_retries}{effort_suffix})..."
        )
        try:
            kwargs: dict[str, Any] = dict(
                model=model,
                max_tokens=budget,
                messages=anthropic_messages,
            )
            if system_payload:
                kwargs["system"] = system_payload
            if resolved_effort:
                kwargs["output_config"] = {"effort": resolved_effort}
            response = client.messages.create(**kwargs)
        except anthropic.RateLimitError as e:
            wait = min(30, 2 ** attempt)
            print(f"[Claude] Rate limit: {e}. Sleeping {wait}s and retrying...")
            time.sleep(wait)
            continue
        except anthropic.APIStatusError as e:
            status = getattr(e, "status_code", None)
            error_text = str(e)
            print(f"[Claude] APIStatusError {status}: {error_text[:300]}")
            normalized = error_text.lower()

            # Context overflow / image size error -> downscale and retry.
            is_too_large = (
                "context" in normalized
                or "prompt is too long" in normalized
                or "request_too_large" in normalized
                or "image" in normalized and "too large" in normalized
            )
            if is_too_large:
                next_edge = _next_smaller_image_edge(current_edge, image_min_edge)
                if next_edge is not None:
                    print(f"[Claude] Retrying with image_max_edge={next_edge}")
                    current_edge = next_edge
                    continue
                if forced_max_images is None or forced_max_images > 1:
                    forced_max_images = 1
                    print("[Claude] Retrying with at most 1 image per request.")
                    continue

            if status and 500 <= int(status) < 600:
                wait = min(30, 2 ** attempt)
                print(f"[Claude] Server error {status}. Sleeping {wait}s and retrying...")
                time.sleep(wait)
                continue

            return None
        except anthropic.APIError as e:
            wait = min(30, 2 ** attempt)
            print(f"[Claude] APIError: {e}. Sleeping {wait}s and retrying...")
            time.sleep(wait)
            continue
        except Exception as e:
            print(f"[Claude] Unexpected error: {type(e).__name__}: {e}")
            return None

        text = _extract_text_from_response(response)
        if text is None:
            stop_reason = getattr(response, "stop_reason", None)
            usage = getattr(response, "usage", None)
            output_tokens = getattr(usage, "output_tokens", None)
            print(
                "[Claude] No text block in response "
                f"(stop_reason={stop_reason!r}, output_tokens={output_tokens!r}, "
                f"budget={budget})."
            )
            if attempt + 1 < max_retries:
                wait = min(5, 2 ** attempt)
                print(
                    "[Claude] Empty text content in response. "
                    f"Sleeping {wait}s and retrying..."
                )
                time.sleep(wait)
                continue
            print("[Claude] Empty text content after final attempt; returning None.")
            return None
        return text

    print("[Claude] Exhausted retries.")
    return None
