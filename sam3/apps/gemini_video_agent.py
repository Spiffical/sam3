
import os
import argparse
import json
import copy
import io
from importlib import resources as importlib_resources
import re
from dataclasses import dataclass
import cv2
import torch
import numpy as np
import time
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

try:
    from google import genai as genai_new
    from google.genai import types as genai_new_types
except Exception:
    genai_new = None
    genai_new_types = None

try:
    import google.generativeai as genai_legacy
except Exception:
    genai_legacy = None

# Load environment variables
load_dotenv()

# Resolve repo root robustly and keep sam3 importable regardless of cwd.
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from sam3.agent.agent_core import agent_inference
from sam3.agent.client_sam3 import sam3_inference, remove_overlapping_masks
from sam3.agent.viz import visualize
from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.apps.interactive_video.backend import PredictorBackend
from nibi_model_compare.run_video_agent_openai import (
    overlay_masks_on_frame,
    save_frame_outputs_json,
)

# -- Gemini Client Adapter --


def find_bpe_path() -> str:
    env_path = os.environ.get("SAM3_BPE_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    candidates = [
        os.path.join(REPO_ROOT, "assets", "bpe_simple_vocab_16e6.txt.gz"),
        os.path.join(REPO_ROOT, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz"),
        "assets/bpe_simple_vocab_16e6.txt.gz",
        "sam3/assets/bpe_simple_vocab_16e6.txt.gz",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    # Fallback when installed as package/editable.
    try:
        resource_path = str(
            importlib_resources.files("sam3").joinpath(
                "assets/bpe_simple_vocab_16e6.txt.gz"
            )
        )
        if os.path.exists(resource_path):
            return resource_path
    except Exception:
        pass

    raise FileNotFoundError(
        f"Could not find bpe_simple_vocab_16e6.txt.gz in: {candidates}"
    )


def _load_image_for_gemini(img_path: str):
    """
    Load image as RGB PIL and optionally downscale to reduce multimodal token load.
    """
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        try:
            max_edge = int(os.environ.get("SAM3_GEMINI_IMAGE_MAX_EDGE", "896"))
        except ValueError:
            max_edge = 896
        if max_edge > 0:
            w, h = img.size
            longest = max(w, h)
            if longest > max_edge:
                scale = max_edge / float(longest)
                new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
        return img.copy()

def _convert_openai_messages_to_gemini(messages: List[Dict[str, Any]]):
    gemini_history = []
    
    for msg in messages:
        # Map roles: user->user, assistant->model. System handled separately.
        if msg["role"] == "system":
            continue
            
        role = "user" if msg["role"] == "user" else "model"
        parts = []
        
        content = msg.get("content", [])
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text = item["text"]
                        # Keep assistant history concise and action-focused to
                        # reduce prompt ambiguity and token load on later rounds.
                        if role == "model":
                            tool_match = re.search(
                                r"<tool>.*?</tool>", text, flags=re.DOTALL
                            )
                            if tool_match:
                                text = tool_match.group(0)
                            else:
                                text = text[-2000:]
                        parts.append(text)
                    elif item.get("type") == "image":
                        # Load image
                        img_path = item["image"]
                        try:
                            # Gemini accepts PIL Image
                            img = _load_image_for_gemini(img_path)
                            parts.append(img)
                        except Exception as e:
                            print(f"[Warn] Could not load image {img_path} for Gemini: {e}")
                            
        if role == "user" and os.environ.get("SAM3_GEMINI_TEXT_BEFORE_IMAGE", "1") == "1":
            text_parts = [p for p in parts if isinstance(p, str)]
            non_text_parts = [p for p in parts if not isinstance(p, str)]
            parts = text_parts + non_text_parts

        if parts:
            gemini_history.append({"role": role, "parts": parts})
            
    return gemini_history

@dataclass
class GeminiClientHandle:
    backend: str  # "new" | "legacy"
    model_name: str
    new_client: Any = None
    legacy_model: Any = None

    def close(self) -> None:
        if self.backend == "new" and self.new_client is not None:
            close_fn = getattr(self.new_client, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass


def _resolve_gemini_sdk() -> str:
    sdk = os.environ.get("SAM3_GEMINI_SDK", "auto").strip().lower()
    if sdk not in {"auto", "new", "legacy"}:
        print(f"[Warn] Invalid SAM3_GEMINI_SDK='{sdk}', falling back to 'auto'.")
        return "auto"
    return sdk


def _make_new_sdk_http_options() -> Optional[Any]:
    if genai_new_types is None:
        return None
    raw = os.environ.get("SAM3_GEMINI_TIMEOUT_SEC", "").strip()
    if not raw:
        return None
    try:
        timeout_sec = float(raw)
    except ValueError:
        print(f"[Warn] Ignoring invalid SAM3_GEMINI_TIMEOUT_SEC='{raw}'")
        return None
    if timeout_sec <= 0:
        return None

    # google-genai HttpOptions.timeout is in milliseconds.
    timeout_ms = int(timeout_sec * 1000.0)
    if timeout_ms <= 0:
        timeout_ms = 1

    # Also pass through client_args timeout in seconds for httpx-level controls.
    client_args = {"timeout": timeout_sec}

    # google-genai exposes timeout through HttpOptions.
    try:
        return genai_new_types.HttpOptions(
            timeout=timeout_ms,
            client_args=client_args,
            async_client_args=client_args,
        )
    except Exception as e:
        print(f"[Warn] Failed to create new SDK HttpOptions: {e}")
        return None


def get_gemini_client(api_key, model_name="gemini-2.5-flash"):
    sdk = _resolve_gemini_sdk()

    if sdk == "new" and genai_new is None:
        raise RuntimeError(
            "SAM3_GEMINI_SDK=new requested but google-genai is not installed. "
            "Install it with: pip install -U google-genai"
        )

    if sdk in {"auto", "new"} and genai_new is not None:
        try:
            http_options = _make_new_sdk_http_options()
            if http_options is not None:
                client = genai_new.Client(api_key=api_key, http_options=http_options)
            else:
                client = genai_new.Client(api_key=api_key)
            print("[Info] Gemini SDK backend: google-genai (new)")
            return GeminiClientHandle(
                backend="new",
                model_name=model_name,
                new_client=client,
            )
        except Exception as e:
            if sdk == "new":
                raise RuntimeError(
                    f"Failed to initialize google-genai client for model '{model_name}': {e}"
                ) from e
            print(
                "[Warn] Failed to initialize google-genai client; "
                f"falling back to legacy SDK. Error: {e}"
            )

    if genai_legacy is None:
        raise RuntimeError(
            "Neither google-genai nor google-generativeai is available. "
            "Install one of them (recommended: pip install -U google-genai)."
        )

    genai_legacy.configure(api_key=api_key)
    try:
        model = genai_legacy.GenerativeModel(model_name)
    except Exception as e:
        print(
            f"[Warn] Failed to initialize legacy model '{model_name}'. "
            f"Falling back to 'gemini-1.5-flash'. Error: {e}"
        )
        model_name = "gemini-1.5-flash"
        model = genai_legacy.GenerativeModel(model_name)

    print("[Info] Gemini SDK backend: google-generativeai (legacy)")
    return GeminiClientHandle(
        backend="legacy",
        model_name=model_name,
        legacy_model=model,
    )


_GEMINI_DEBUG_COUNTER = 0


def _next_gemini_debug_trace_id() -> str:
    global _GEMINI_DEBUG_COUNTER
    _GEMINI_DEBUG_COUNTER += 1
    return f"{int(time.time())}_{_GEMINI_DEBUG_COUNTER:06d}"


def _get_gemini_debug_dir() -> Optional[str]:
    debug_dir = os.environ.get("SAM3_GEMINI_DEBUG_DIR", "").strip()
    if not debug_dir:
        return None
    os.makedirs(debug_dir, exist_ok=True)
    return debug_dir


def _summarize_part_for_debug(part: Any) -> Dict[str, Any]:
    if isinstance(part, str):
        return {
            "type": "text",
            "chars": len(part),
            "preview": part[:300],
        }
    if isinstance(part, Image.Image):
        return {
            "type": "image",
            "size": [part.width, part.height],
            "mode": part.mode,
        }
    return {"type": type(part).__name__}


def _summarize_messages_for_debug(
    gemini_messages: List[Dict[str, Any]], system_prompt: str
) -> Dict[str, Any]:
    summary_messages = []
    for msg in gemini_messages:
        parts = msg.get("parts", [])
        summary_messages.append(
            {
                "role": msg.get("role"),
                "num_parts": len(parts),
                "parts": [_summarize_part_for_debug(p) for p in parts],
            }
        )
    return {
        "system_prompt_chars": len(system_prompt),
        "num_messages": len(gemini_messages),
        "messages": summary_messages,
    }


def _dump_gemini_debug_json(
    trace_id: str, label: str, payload: Dict[str, Any]
) -> None:
    if os.environ.get("SAM3_GEMINI_DEBUG_DUMP", "0") != "1":
        return
    debug_dir = _get_gemini_debug_dir()
    if debug_dir is None:
        return
    out_path = os.path.join(debug_dir, f"{trace_id}_{label}.json")
    try:
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
    except Exception as e:
        print(f"[Debug] Failed to write Gemini debug artifact {out_path}: {e}")


def _response_to_json_like(response: Any) -> Dict[str, Any]:
    if response is None:
        return {"response": None}
    try:
        return response.to_dict()
    except Exception:
        try:
            return response.model_dump()
        except Exception:
            pass
        try:
            return dict(response)
        except Exception:
            pass
        try:
            return {"response_repr": repr(response)}
        except Exception:
            return {"response_repr": "<unavailable>"}


def _make_request_options() -> Optional[Dict[str, Any]]:
    raw = os.environ.get("SAM3_GEMINI_TIMEOUT_SEC", "").strip()
    if not raw:
        return None
    try:
        timeout_sec = float(raw)
    except ValueError:
        print(f"[Warn] Ignoring invalid SAM3_GEMINI_TIMEOUT_SEC='{raw}'")
        return None
    if timeout_sec <= 0:
        return None
    return {"timeout": timeout_sec}


def _should_use_native_system_instruction() -> bool:
    return os.environ.get("SAM3_GEMINI_USE_SYSTEM_INSTRUCTION", "1") == "1"


def _build_request_model(
    model_name: str,
    system_prompt: str,
    fallback_model: Any,
) -> Tuple[Any, bool]:
    """
    Return (model_for_request, uses_native_system_instruction).
    """
    if not system_prompt or not _should_use_native_system_instruction():
        return fallback_model, False

    if isinstance(fallback_model, GeminiClientHandle) and fallback_model.backend == "new":
        # google-genai supports system_instruction via GenerateContentConfig.
        return fallback_model, True

    try:
        model = genai_legacy.GenerativeModel(
            model_name,
            system_instruction=system_prompt,
        )
        if isinstance(fallback_model, GeminiClientHandle):
            return (
                GeminiClientHandle(
                    backend="legacy",
                    model_name=model_name,
                    legacy_model=model,
                ),
                True,
            )
        return model, True
    except Exception as e:
        print(
            "[Warn] Failed to apply system_instruction natively; using inline prompt. "
            f"Error: {e}"
        )
        return fallback_model, False


def _resolve_api_mode() -> str:
    mode = os.environ.get("SAM3_GEMINI_API_MODE", "auto").strip().lower()
    if mode not in {"auto", "chat", "generate"}:
        print(f"[Warn] Invalid SAM3_GEMINI_API_MODE='{mode}', falling back to 'auto'.")
        return "auto"
    return mode


def _extract_text_from_gemini_response(response):
    """Best-effort text extraction across Gemini SDK response shapes."""
    # Fast path.
    try:
        txt = response.text
        if txt and str(txt).strip():
            return str(txt).strip()
    except Exception:
        pass

    # Fallback path for responses where .text accessor fails.
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) if content is not None else None
        if not parts:
            continue
        chunks = []
        for part in parts:
            part_text = getattr(part, "text", None)
            if part_text and str(part_text).strip():
                chunks.append(str(part_text).strip())
                continue
            if isinstance(part, dict):
                dict_text = part.get("text")
                if dict_text and str(dict_text).strip():
                    chunks.append(str(dict_text).strip())
        merged = "\n".join(chunks).strip()
        if merged:
            return merged

    return None


def _debug_print_gemini_response(response):
    """Concise diagnostics for empty/invalid Gemini outputs."""
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        print("[Debug] Gemini response has no candidates.")
    else:
        first = candidates[0]
        finish_reason = getattr(first, "finish_reason", None)
        finish_message = getattr(first, "finish_message", None)
        content = getattr(first, "content", None)
        parts = getattr(content, "parts", None) if content is not None else None
        num_parts = len(parts) if parts is not None else 0
        print(f"[Debug] Finish Reason: {finish_reason}")
        if finish_message:
            print(f"[Debug] Finish Message: {finish_message}")
        print(f"[Debug] Num Parts: {num_parts}")
        print(f"[Debug] Safety Ratings: {getattr(first, 'safety_ratings', None)}")
    print(f"[Debug] Prompt Feedback: {getattr(response, 'prompt_feedback', None)}")
    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
        try:
            usage_dict = (
                usage.to_dict()
                if hasattr(usage, "to_dict")
                else dict(usage)
            )
        except Exception:
            usage_dict = str(usage)
        print(f"[Debug] Usage Metadata: {usage_dict}")

    # Optional verbose dump for deeper local debugging.
    if os.environ.get("SAM3_GEMINI_DEBUG_RESPONSE", "0") == "1":
        try:
            if hasattr(response, "to_dict"):
                payload = response.to_dict()
            elif hasattr(response, "model_dump"):
                payload = response.model_dump()
            else:
                payload = {"response_repr": repr(response)}
            print("[Debug] Raw response JSON:")
            print(json.dumps(payload, indent=2))
        except Exception as e:
            print(f"[Debug] Could not serialize raw response: {e}")


def _make_generation_config():
    # Keep legacy behavior unless explicitly enabled.
    if os.environ.get("SAM3_GEMINI_USE_GENERATION_CONFIG", "0") != "1":
        return None

    try:
        max_output_tokens = int(
            os.environ.get("SAM3_GEMINI_MAX_OUTPUT_TOKENS", "20000")
        )
    except ValueError:
        max_output_tokens = 20000
    try:
        temperature = float(os.environ.get("SAM3_GEMINI_TEMPERATURE", "0.2"))
    except ValueError:
        temperature = 0.2

    # google.generativeai accepts either dict or GenerationConfig.
    return {
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
    }


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _default_new_sdk_max_output_tokens() -> int:
    raw = os.environ.get("SAM3_GEMINI_MAX_OUTPUT_TOKENS", "").strip()
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    # Keep a sane default for tool-call style outputs.
    return 4096


def _make_new_generate_content_config(
    *,
    generation_config: Optional[Dict[str, Any]],
    system_prompt: str,
    use_native_system_instruction: bool,
) -> Optional[Any]:
    cfg: Dict[str, Any] = {}
    if generation_config is not None:
        cfg.update(generation_config)
    if use_native_system_instruction and system_prompt:
        cfg["system_instruction"] = system_prompt

    # Reliability defaults for new SDK to avoid empty STOP responses.
    if _env_bool("SAM3_GEMINI_NEW_FORCE_TEXT_CONFIG", True):
        cfg.setdefault("candidate_count", 1)
        cfg.setdefault("max_output_tokens", _default_new_sdk_max_output_tokens())
        cfg.setdefault("response_mime_type", "text/plain")
        cfg.setdefault("response_modalities", ["TEXT"])

    if _env_bool("SAM3_GEMINI_DISABLE_THINKING", True):
        cfg.setdefault(
            "thinking_config",
            {
                "thinking_budget": 0,
                "include_thoughts": False,
            },
        )

    if not cfg:
        return None

    if genai_new_types is None:
        return cfg

    try:
        return genai_new_types.GenerateContentConfig(**cfg)
    except Exception as e:
        # Keep moving with raw dict config if typed config construction fails.
        print(f"[Warn] Failed to build typed GenerateContentConfig: {e}")
        return cfg


def _new_sdk_part_from_value(value: Any) -> Any:
    if genai_new_types is None:
        return value

    # Already a typed Part.
    if isinstance(value, genai_new_types.Part):
        return value

    if isinstance(value, str):
        return genai_new_types.Part.from_text(text=value)

    if isinstance(value, Image.Image):
        buf = io.BytesIO()
        value.convert("RGB").save(buf, format="JPEG", quality=90, optimize=True)
        return genai_new_types.Part.from_bytes(
            data=buf.getvalue(),
            mime_type="image/jpeg",
        )

    if isinstance(value, dict):
        # Common shorthand.
        if "text" in value and isinstance(value.get("text"), str):
            return genai_new_types.Part.from_text(text=value["text"])
        # Attempt to parse as a Part dict.
        try:
            return genai_new_types.Part(**value)
        except Exception:
            # Last resort: stringify for robustness.
            return genai_new_types.Part.from_text(text=str(value))

    # Keep request flowing even for unexpected objects.
    return genai_new_types.Part.from_text(text=str(value))


def _normalize_contents_for_new_sdk(contents: Any) -> Any:
    if genai_new_types is None:
        return contents

    # If this is message-history style: [{"role": "...", "parts": [...]}, ...]
    if isinstance(contents, list) and contents and isinstance(contents[0], dict):
        first = contents[0]
        if "role" in first and "parts" in first:
            normalized_messages = []
            for message in contents:
                role = str(message.get("role", "user")).lower()
                if role not in {"user", "model"}:
                    role = "user"
                raw_parts = message.get("parts", [])
                if not isinstance(raw_parts, list):
                    raw_parts = [raw_parts]
                parts = [_new_sdk_part_from_value(p) for p in raw_parts]
                normalized_messages.append(
                    genai_new_types.Content(role=role, parts=parts)
                )
            return normalized_messages

    # Otherwise treat as a single-turn list of parts.
    if isinstance(contents, list):
        return [_new_sdk_part_from_value(p) for p in contents]

    # Scalar single-turn content.
    return _new_sdk_part_from_value(contents)


def _call_generate_content(
    model: Any,
    contents: Any,
    generation_config: Optional[Dict[str, Any]],
    request_options: Optional[Dict[str, Any]],
    system_prompt: str,
    use_native_system_instruction: bool,
):
    if (
        isinstance(model, GeminiClientHandle)
        and model.backend == "new"
        and model.new_client is not None
    ):
        if request_options is not None and request_options.get("timeout") is not None:
            # Timeout is best configured through Client(http_options=...) in google-genai.
            pass
        config = _make_new_generate_content_config(
            generation_config=generation_config,
            system_prompt=system_prompt,
            use_native_system_instruction=use_native_system_instruction,
        )
        normalized_contents = _normalize_contents_for_new_sdk(contents)
        return model.new_client.models.generate_content(
            model=model.model_name,
            contents=normalized_contents,
            config=config,
        )

    legacy_model = (
        model.legacy_model
        if isinstance(model, GeminiClientHandle) and model.legacy_model is not None
        else model
    )
    kwargs = {}
    if generation_config is not None:
        kwargs["generation_config"] = generation_config
    if request_options is not None:
        kwargs["request_options"] = request_options
    return legacy_model.generate_content(contents, **kwargs)


def _call_chat_send_message(
    model: Any,
    history: List[Dict[str, Any]],
    last_message_parts: List[Any],
    generation_config: Optional[Dict[str, Any]],
    request_options: Optional[Dict[str, Any]],
    system_prompt: str,
    use_native_system_instruction: bool,
):
    # For new SDK, use generate_content with explicit full history to keep behavior
    # consistent with our multimodal prompt-building logic.
    if isinstance(model, GeminiClientHandle) and model.backend == "new":
        contents = list(history) + [{"role": "user", "parts": last_message_parts}]
        return _call_generate_content(
            model=model,
            contents=contents,
            generation_config=generation_config,
            request_options=request_options,
            system_prompt=system_prompt,
            use_native_system_instruction=use_native_system_instruction,
        )

    legacy_model = (
        model.legacy_model
        if isinstance(model, GeminiClientHandle) and model.legacy_model is not None
        else model
    )
    chat = legacy_model.start_chat(history=history)
    kwargs = {}
    if generation_config is not None:
        kwargs["generation_config"] = generation_config
    if request_options is not None:
        kwargs["request_options"] = request_options
    return chat.send_message(last_message_parts, **kwargs)


def _recovery_generate_request(
    model,
    gemini_messages,
    generation_config: Optional[Dict[str, Any]],
    request_options: Optional[Dict[str, Any]],
    system_prompt: str,
    use_native_system_instruction: bool,
):
    """
    Last-resort recovery call when Gemini returns STOP with empty content.
    Uses a minimal, explicit instruction and only the most relevant context.
    """
    if not gemini_messages:
        return None

    # Prefer latest user message; also include first user message for raw image/query.
    first_user = next((m for m in gemini_messages if m.get("role") == "user"), None)
    last_user = None
    for m in reversed(gemini_messages):
        if m.get("role") == "user":
            last_user = m
            break

    recovery_parts = [
        (
            "Return exactly one valid tool call and nothing else in this format: "
            '<tool>{"name":"TOOL_NAME","parameters":{...}}</tool>. '
            "Allowed tool names: segment_phrase, examine_each_mask, "
            "select_masks_and_return, report_no_mask."
        )
    ]
    if first_user is not None:
        recovery_parts.extend(first_user.get("parts", []))
    if last_user is not None and last_user is not first_user:
        recovery_parts.extend(last_user.get("parts", []))

    try:
        return _call_generate_content(
            model=model,
            contents=recovery_parts,
            generation_config=generation_config,
            request_options=request_options,
            system_prompt=system_prompt,
            # Keep recovery call minimal: do not attach full system instruction.
            use_native_system_instruction=False,
        )
    except Exception as e:
        print(f"[Warn] Recovery generate_content failed: {e}")
        return None


def _try_request_modes_with_retries(
    *,
    trace_id: str,
    request_model: Any,
    gemini_messages: List[Dict[str, Any]],
    generation_config: Optional[Dict[str, Any]],
    request_options: Optional[Dict[str, Any]],
    api_mode: str,
    max_retries: int,
    label_prefix: str,
    system_prompt: str,
    use_native_system_instruction: bool,
) -> Optional[str]:
    for attempt in range(max_retries):
        try:
            # Add a small delay to be polite and avoid burst limits
            if attempt == 0:
                time.sleep(2)

            history = gemini_messages[:-1]
            last_msg = gemini_messages[-1]

            # Try generate_content first (more stable in some multimodal cases).
            if api_mode in {"auto", "generate"}:
                generate_response = _call_generate_content(
                    request_model,
                    gemini_messages,
                    generation_config=generation_config,
                    request_options=request_options,
                    system_prompt=system_prompt,
                    use_native_system_instruction=use_native_system_instruction,
                )
                _dump_gemini_debug_json(
                    trace_id,
                    f"{label_prefix}_attempt_{attempt+1:02d}_generate_response",
                    _response_to_json_like(generate_response),
                )
                generate_text = _extract_text_from_gemini_response(generate_response)
                if generate_text:
                    return generate_text
                print(
                    f"[Error] Gemini returned no usable text in generate_content mode "
                    f"(Attempt {attempt+1}/{max_retries})."
                )
                _debug_print_gemini_response(generate_response)

            # Then try chat mode.
            if api_mode in {"auto", "chat"}:
                chat_response = _call_chat_send_message(
                    request_model,
                    history=history,
                    last_message_parts=last_msg["parts"],
                    generation_config=generation_config,
                    request_options=request_options,
                    system_prompt=system_prompt,
                    use_native_system_instruction=use_native_system_instruction,
                )
                _dump_gemini_debug_json(
                    trace_id,
                    f"{label_prefix}_attempt_{attempt+1:02d}_chat_response",
                    _response_to_json_like(chat_response),
                )
                chat_text = _extract_text_from_gemini_response(chat_response)
                if chat_text:
                    return chat_text
                print(
                    f"[Error] Gemini returned no usable text in chat mode "
                    f"(Attempt {attempt+1}/{max_retries})."
                )
                _debug_print_gemini_response(chat_response)

            # Last-resort recovery call with minimal context and strict format.
            recovery_response = _recovery_generate_request(
                request_model,
                gemini_messages,
                generation_config=generation_config,
                request_options=request_options,
                system_prompt=system_prompt,
                use_native_system_instruction=use_native_system_instruction,
            )
            _dump_gemini_debug_json(
                trace_id,
                f"{label_prefix}_attempt_{attempt+1:02d}_recovery_response",
                _response_to_json_like(recovery_response),
            )
            recovery_text = _extract_text_from_gemini_response(recovery_response)
            if recovery_text:
                print("[Info] Recovered response via strict recovery request.")
                return recovery_text
            if recovery_response is not None:
                print("[Warn] Recovery request also returned no usable text.")
                _debug_print_gemini_response(recovery_response)

            time.sleep(5)
            continue

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "Resource has been exhausted" in err_str:
                sleep_time = 20 * (attempt + 1)
                print(f"[Warn] Rate limit hit. Sleeping for {sleep_time} seconds before retry...")
                time.sleep(sleep_time)
                continue
            else:
                print(f"Gemini Request failed: {e}")
                _dump_gemini_debug_json(
                    trace_id,
                    f"{label_prefix}_attempt_{attempt+1:02d}_exception",
                    {"error": err_str},
                )
                time.sleep(5)
                continue
    return None


def gemini_send_request(messages, model):
    """
    Adapter function to match signature expected by agent_inference.
    messages: list of dicts (OpenAI style)
    """
    trace_id = _next_gemini_debug_trace_id()

    # Extract system prompt
    system_prompt = ""
    for msg in messages:
        if msg["role"] == "system":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join([x["text"] for x in content if x.get("type") == "text"])
            system_prompt += str(content) + "\n\n"

    model_name = (
        getattr(model, "model_name", None)
        or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    )
    request_model, using_native_system_instruction = _build_request_model(
        model_name=model_name,
        system_prompt=system_prompt.strip(),
        fallback_model=model,
    )

    gemini_messages = _convert_openai_messages_to_gemini(messages)

    if not gemini_messages:
        return None

    # If native system_instruction is unavailable, inline the prompt as first user part.
    if system_prompt and not using_native_system_instruction:
        if gemini_messages[0]["role"] == "user":
            gemini_messages[0]["parts"].insert(0, system_prompt)

    _dump_gemini_debug_json(
        trace_id,
        "request",
        {
            "model_name": model_name,
            "sdk_backend": getattr(request_model, "backend", "legacy-object"),
            "using_native_system_instruction": using_native_system_instruction,
            "api_mode": _resolve_api_mode(),
            "new_sdk_force_text_config": _env_bool(
                "SAM3_GEMINI_NEW_FORCE_TEXT_CONFIG", True
            ),
            "new_sdk_disable_thinking": _env_bool(
                "SAM3_GEMINI_DISABLE_THINKING", True
            ),
            "generation_config": _make_generation_config(),
            "request_options": _make_request_options(),
            "messages_summary": _summarize_messages_for_debug(
                gemini_messages=gemini_messages,
                system_prompt=system_prompt,
            ),
        },
    )

    max_retries = 5
    generation_config = _make_generation_config()
    request_options = _make_request_options()
    api_mode = _resolve_api_mode()
    text = _try_request_modes_with_retries(
        trace_id=trace_id,
        request_model=request_model,
        gemini_messages=gemini_messages,
        generation_config=generation_config,
        request_options=request_options,
        api_mode=api_mode,
        max_retries=max_retries,
        label_prefix="native" if using_native_system_instruction else "inline",
        system_prompt=system_prompt.strip(),
        use_native_system_instruction=using_native_system_instruction,
    )
    if text:
        return text

    # Retry with the exact same full prompt inlined if native system_instruction
    # path yielded empty outputs.
    if system_prompt and using_native_system_instruction:
        print(
            "[Warn] Empty Gemini outputs with native system_instruction. "
            "Retrying with the same full system prompt inlined."
        )
        inline_messages = copy.deepcopy(gemini_messages)
        if inline_messages and inline_messages[0].get("role") == "user":
            inline_messages[0]["parts"] = [system_prompt] + inline_messages[0].get(
                "parts", []
            )
        _dump_gemini_debug_json(
            trace_id,
            "inline_retry_request",
            {
                "model_name": model_name,
                "using_native_system_instruction": False,
                "api_mode": api_mode,
                "generation_config": generation_config,
                "request_options": request_options,
                "messages_summary": _summarize_messages_for_debug(
                    gemini_messages=inline_messages,
                    system_prompt=system_prompt,
                ),
            },
        )
        text = _try_request_modes_with_retries(
            trace_id=trace_id,
            request_model=model,
            gemini_messages=inline_messages,
            generation_config=generation_config,
            request_options=request_options,
            api_mode=api_mode,
            max_retries=max_retries,
            label_prefix="inline_retry",
            system_prompt=system_prompt.strip(),
            use_native_system_instruction=False,
        )
        if text:
            return text

    return None

# -- SAM3 Service Adapter --

class LocalSam3Service:
    def __init__(self, processor, output_dir):
        self.processor = processor
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def call_service(self, image_path, text_prompt, output_folder_path=None):
        if output_folder_path is None:
            output_folder_path = self.output_dir
            
        print(f"Call SAM3 Service: {image_path}, prompt={text_prompt}")
        
        try:
            # Reusing code from sam3.agent.client_sam3.call_sam_service logic
            # but invoking local processor
            
            # 1. Run inference
            outputs = sam3_inference(self.processor, image_path, text_prompt)
            
            # 2. Cleanup
            outputs = remove_overlapping_masks(outputs)
            
            safe_prompt = text_prompt.replace("/", "_").replace(" ", "_")
            out_name = f"{os.path.basename(image_path)}_{safe_prompt}"
            
            output_image_path = os.path.join(output_folder_path, f"{out_name}.png")
            output_json_path = os.path.join(output_folder_path, f"{out_name}.json")
            
            outputs = {
                "original_image_path": image_path,
                "output_image_path": output_image_path,
                **outputs
            }
            
            # Sort by scores
            if "pred_scores" in outputs and outputs["pred_scores"]:
                score_indices = sorted(
                    range(len(outputs["pred_scores"])),
                    key=lambda i: outputs["pred_scores"][i],
                    reverse=True,
                )
                outputs["pred_scores"] = [outputs["pred_scores"][i] for i in score_indices]
                outputs["pred_boxes"] = [outputs["pred_boxes"][i] for i in score_indices]
                outputs["pred_masks"] = [outputs["pred_masks"][i] for i in score_indices]

            # Filter short masks
            valid_masks = []
            valid_boxes = []
            valid_scores = []
            for i, rle in enumerate(outputs["pred_masks"]):
                if len(rle) > 4:
                    valid_masks.append(rle)
                    valid_boxes.append(outputs["pred_boxes"][i])
                    valid_scores.append(outputs["pred_scores"][i])
            outputs["pred_masks"] = valid_masks
            outputs["pred_boxes"] = valid_boxes
            outputs["pred_scores"] = valid_scores
            
            # Save JSON
            with open(output_json_path, "w") as f:
                json.dump(outputs, f, indent=4)
                
            # Render
            viz = visualize(outputs)
            viz.save(output_image_path)
            
            return output_json_path
            
        except Exception as e:
            print(f"Error in SAM3 service: {e}")
            raise e

# -- Main Video Runner --

def mask_to_points(mask, num_points=1):
    """
    Convert a binary mask to a set of points inside the mask.
    Simple method: find coordinates where mask is True and random sample.
    """
    y_indices, x_indices = np.where(mask)
    if len(y_indices) == 0:
        return []
    
    # Select random points
    coords = list(zip(x_indices, y_indices))
    import random
    if len(coords) > num_points:
        return random.sample(coords, num_points)
    return coords

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="Path to video file")
    parser.add_argument("--prompt", type=str, default="Identify and segment any biological creatures.", help="Initial prompt for Agent")
    parser.add_argument("--api_key", type=str, help="Gemini API Key")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Gemini model name (CLI takes precedence over GEMINI_MODEL).",
    )
    parser.add_argument("--output_dir", type=str, default="sam3_video_agent_out")
    parser.add_argument("--gpus", type=str, default="0", help="GPUs to use")
    parser.add_argument(
        "--gemini_api_mode",
        type=str,
        choices=["auto", "generate", "chat"],
        default=None,
        help="Gemini call strategy (CLI takes precedence over SAM3_GEMINI_API_MODE).",
    )
    parser.add_argument(
        "--gemini_sdk",
        type=str,
        choices=["auto", "new", "legacy"],
        default=None,
        help="Gemini Python SDK backend selection (CLI takes precedence over SAM3_GEMINI_SDK).",
    )
    parser.add_argument(
        "--gemini_timeout_sec",
        type=float,
        default=None,
        help="Optional timeout in seconds for Gemini API calls.",
    )
    parser.add_argument(
        "--gemini_use_system_instruction",
        type=int,
        choices=[0, 1],
        default=None,
        help=(
            "Use Gemini native system_instruction support (CLI takes precedence "
            "over SAM3_GEMINI_USE_SYSTEM_INSTRUCTION)."
        ),
    )
    parser.add_argument(
        "--gemini_debug_dump",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Dump Gemini request/response artifacts to JSON files "
            "(CLI takes precedence over SAM3_GEMINI_DEBUG_DUMP)."
        ),
    )
    parser.add_argument(
        "--gemini_debug_dir",
        type=str,
        default=None,
        help=(
            "Directory for Gemini debug artifacts "
            "(CLI takes precedence over SAM3_GEMINI_DEBUG_DIR)."
        ),
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=1008,
        help="Video predictor processing size. For current SAM3 checkpoint, use 1008.",
    )
    parser.add_argument(
        "--analysis_second",
        type=float,
        default=0.0,
        help="Which second of the video to sample for agent analysis (default: 0.0).",
    )
    parser.add_argument(
        "--save_frame_outputs_json",
        action="store_true",
        help="Save propagated per-frame outputs (obj IDs, boxes, masks as RLE) to JSON.",
    )
    parser.add_argument(
        "--frame_outputs_json_path",
        type=str,
        default="",
        help="Optional explicit path for per-frame outputs JSON.",
    )
    parser.add_argument("--save_prompts", action="store_true", help="Save prompts to JSON and exit without propagation")

    args = parser.parse_args()

    # Resolve config with CLI-first precedence, env fallback second.
    resolved_model = args.model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    resolved_api_mode = args.gemini_api_mode or os.environ.get(
        "SAM3_GEMINI_API_MODE", "auto"
    )
    resolved_gemini_sdk = args.gemini_sdk or os.environ.get(
        "SAM3_GEMINI_SDK", "auto"
    )
    resolved_use_system_instruction = (
        args.gemini_use_system_instruction
        if args.gemini_use_system_instruction is not None
        else int(os.environ.get("SAM3_GEMINI_USE_SYSTEM_INSTRUCTION", "1"))
    )
    resolved_debug_dump = (
        args.gemini_debug_dump
        if args.gemini_debug_dump is not None
        else os.environ.get("SAM3_GEMINI_DEBUG_DUMP", "0") == "1"
    )
    resolved_debug_dir = (
        args.gemini_debug_dir
        if args.gemini_debug_dir is not None
        else os.environ.get("SAM3_GEMINI_DEBUG_DIR", "")
    )
    if resolved_debug_dump and not resolved_debug_dir:
        resolved_debug_dir = os.path.join(args.output_dir, "gemini_debug")

    # Propagate resolved values to environment so lower-level helpers use one source.
    os.environ["SAM3_GEMINI_API_MODE"] = resolved_api_mode
    os.environ["SAM3_GEMINI_SDK"] = resolved_gemini_sdk
    os.environ["SAM3_GEMINI_USE_SYSTEM_INSTRUCTION"] = str(
        int(resolved_use_system_instruction)
    )
    if args.gemini_timeout_sec is not None:
        os.environ["SAM3_GEMINI_TIMEOUT_SEC"] = str(args.gemini_timeout_sec)
    os.environ["SAM3_GEMINI_DEBUG_DUMP"] = "1" if resolved_debug_dump else "0"
    if resolved_debug_dir:
        os.environ["SAM3_GEMINI_DEBUG_DIR"] = resolved_debug_dir

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Please provide --api_key or set GEMINI_API_KEY/GOOGLE_API_KEY env var")
        return

    # 1. Setup Models
    print("Loading SAM3 Image Model (for Agent)...")
    
    try:
        bpe_path = find_bpe_path()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print(f"Using BPE path: {bpe_path}")
        
    image_model = build_sam3_image_model(bpe_path=bpe_path)
    image_processor = Sam3Processor(image_model, confidence_threshold=0.4)
    local_service = LocalSam3Service(image_processor, os.path.join(args.output_dir, "sam_service"))
    
    print(f"Loading Gemini model: {resolved_model} (sdk={resolved_gemini_sdk})")
    gemini_model = get_gemini_client(api_key, model_name=resolved_model)
    
    # 2. Extract analysis frame
    cap = cv2.VideoCapture(args.video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        print("Failed to determine total frames from video")
        cap.release()
        return

    if fps <= 0:
        # Fallback for malformed metadata.
        fps = 30.0

    requested_second = max(0.0, float(args.analysis_second))
    analysis_frame_idx = int(round(requested_second * fps))
    analysis_frame_idx = max(0, min(total_frames - 1, analysis_frame_idx))
    effective_second = analysis_frame_idx / fps if fps > 0 else 0.0

    cap.set(cv2.CAP_PROP_POS_FRAMES, analysis_frame_idx)
    ret, frame = cap.read()
    if not ret:
        print(
            f"Failed to read frame at second={requested_second:.3f} "
            f"(frame_index={analysis_frame_idx})"
        )
        return
    cap.release()
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_0_path = os.path.join(args.output_dir, f"frame_{analysis_frame_idx}.jpg")
    os.makedirs(args.output_dir, exist_ok=True)
    Image.fromarray(frame_rgb).save(frame_0_path)
    print(
        "Extracted analysis frame "
        f"(requested_second={requested_second:.3f}, "
        f"effective_second={effective_second:.3f}, frame_index={analysis_frame_idx}) "
        f"to {frame_0_path}"
    )
    
    # 3. Run Agent Inference on Frame 0
    print("Running Agent Inference on Frame 0...")
    
    from functools import partial
    send_req = partial(gemini_send_request, model=gemini_model)
    call_sam = local_service.call_service
    
    try:
        history, final_outputs, rendered_img = agent_inference(
             img_path=frame_0_path,
             initial_text_prompt=args.prompt,
             send_generate_request=send_req,
             call_sam_service=call_sam,
             output_dir=os.path.join(args.output_dir, "agent_out"),
             debug=True
        )
    except Exception as e:
        print(f"Agent inference failed: {e}")
        return
    
    print("Agent inference complete.")
    
    selected_masks_rle = final_outputs.get("pred_masks", [])
    print(f"Agent selected {len(selected_masks_rle)} masks.")
    
    if not selected_masks_rle:
        print("No masks found by agent. Exiting video propagation.")
        return

    # 4. Initialize Video Predictor (Only if NOT saving prompts only)
    backend = None
    if not args.save_prompts:
        print("Initializing Video Predictor...")
        gpu_ids = [int(x) for x in args.gpus.split(",")] # Use args.gpus
        backend = PredictorBackend(gpu_ids=gpu_ids)
        
        # Start session
        img0 = cv2.imread(frame_0_path)
        height, width = img0.shape[:2]
        image_size = args.image_size
        
        try:
            session_id = backend.start_session(
                resource_path=args.video_path,
                image_size=image_size
            )
            print(f"Video Session Started: {session_id}")
        except Exception as e:
            print(f"Failed to start video session: {e}")
            return
    else:
        # Load frame 0 to get dimensions for creating prompts
        img0 = cv2.imread(frame_0_path)
        height, width = img0.shape[:2]
        print("Skipping Video Predictor initialization (save_prompts mode).")

    # Convert Agent Masks to Point Prompts for Video Predictor
    # The Agent returns RLE masks.
    # We need to turn them into prompts.
    # We could use the mask directly if PredictorBackend supports it, but add_point_prompt is the main API.
    # We can sample points from the mask.
    
    from pycocotools import mask as mask_util
    
    def decode_rle_to_mask(rle, h, w):
        # Debug
        # print(f"Debugging RLE: {type(rle)}")
        if isinstance(rle, str):
            # Probably just the counts string
            rle = {
                'counts': rle.encode('utf-8'),
                'size': [h, w]
            }
        elif isinstance(rle, dict) and 'counts' in rle:
            counts = rle['counts']
            if isinstance(counts, str):
                try:
                    rle['counts'] = counts.encode('utf-8')
                except Exception as e:
                    print(f"Error encoding counts: {e}")
        
        # mask_util.decode often prefers a list of RLEs
        try:
            m = mask_util.decode([rle])
            # m shape is (H, W, 1)
            return m[:, :, 0]
        except Exception as e:
            print(f"Decode failed: {e}. RLE: {rle}")
            # Try without list?
            m = mask_util.decode(rle)
            return m

    print("Processing masks to points...")
    
    # helper for finding positive point in mask
    def get_center_point(mask):
        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) > 0:
            # Simple centroid or just middle point
            # Let's pick a random point or centroid
            idx = len(y_indices) // 2
            return (x_indices[idx], y_indices[idx])
        return None

    # Collect prompts for storage or propagation
    generated_prompts = []
    
    # Process each mask
    for i, rle in enumerate(selected_masks_rle):
        try:
            mask_array = decode_rle_to_mask(rle, height, width)
            # Handle if mask_array is 3D (H, W, 1) or 2D
            if len(mask_array.shape) == 3:
                mask_array = mask_array[:, :, 0]
        except Exception as e:
            print(f"Skipping mask {i+1} due to decode error: {e}")
            continue
            
        point = get_center_point(mask_array)
        if point:
            x, y = point
            # Points format: [(x, y, label)]
            prompt_data = {
                "frame_idx": int(analysis_frame_idx),
                "obj_id": i + 1,
                "points": [[float(x), float(y), 1]], # 1=positive
                "label": 1,
                "source": "agent"
            }
            generated_prompts.append(prompt_data)
            
            if backend:
                print(f"Adding prompt for object {i+1} at {x}, {y}")
                backend.add_point_prompt(
                    session_id=session_id,
                    frame_idx=int(analysis_frame_idx),
                    obj_id=i+1,
                    points=[(float(x), float(y), 1)],
                    # PredictorBackend expects frame_size as (width, height).
                    frame_size=(width, height)
                )
        else:
            print(f"Could not find valid point for mask {i+1}")
            
    if args.save_prompts:
        # Save prompts to JSON and exit
        # Intelligent location: assets/annotations
        # Filename: [video_stem]_prompts.json
        
        video_stem = os.path.splitext(os.path.basename(args.video_path))[0]
        json_filename = f"{video_stem}_prompts.json"
        
        # We assume assets/annotations is the central store
        # But we should respect if the user is working in a completely different dir?
        # For this project, assets/annotations is the convention requested.
        save_dir = os.path.join("assets", "annotations")
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, json_filename)
        
        with open(save_path, 'w') as f:
            json.dump(generated_prompts, f, indent=2)
            
        print(f"Prompts saved to {save_path}")
        print("Exiting as --save_prompts was specified.")
        return

    # Propagate
    if backend:
        print("Running Propagation...")  
    # 6. Propagate
    print("Propagating masks...")
    prop_gen = backend.propagate({"session_id": session_id, "type": "propagate_in_video"})
    
    # Consume generator
    import tqdm
    results = {}
    for output in tqdm.tqdm(prop_gen):
        # output is dictionary with frame_index, outputs
        f_idx = output["frame_index"]
        results[f_idx] = output["outputs"]
    
    print("Propagation complete.")

    should_save_frame_outputs = args.save_frame_outputs_json or os.environ.get(
        "SAM3_SAVE_FRAME_OUTPUTS_JSON", "1"
    ) == "1"
    frame_outputs_json_path = (
        args.frame_outputs_json_path
        or os.path.join(args.output_dir, "frame_outputs_rle.json")
    )
    if should_save_frame_outputs:
        save_frame_outputs_json(
            frame_outputs_json_path,
            results,
            frame_h=height,
            frame_w=width,
        )
        print(f"Per-frame outputs saved to {frame_outputs_json_path}")
    
    # 7. Render Video
    print("Rendering output video...")
    out_video_path = os.path.join(args.output_dir, "output_video.mp4")
    
    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx in results:
            frame = overlay_masks_on_frame(frame, results[frame_idx])
        
        writer.write(frame)
        frame_idx += 1
        
    cap.release()
    writer.release()
    print(f"Video saved to {out_video_path}")
    backend.shutdown()
    gemini_model.close()

if __name__ == "__main__":
    main()
