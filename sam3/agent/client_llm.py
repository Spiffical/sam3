# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import base64
import io
import os
import re
from typing import Any, Optional

from openai import OpenAI

try:
    from PIL import Image
except Exception:
    Image = None


def get_image_base64_and_mime(image_path, max_edge: Optional[int] = None):
    """Convert image file to base64 string and get MIME type"""
    try:
        # Get MIME type based on file extension
        ext = os.path.splitext(image_path)[1].lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        mime_type = mime_types.get(ext, "image/jpeg")  # Default to JPEG

        # Optionally downscale large images to reduce multimodal token load.
        if Image is not None and max_edge and max_edge > 0:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                width, height = img.size
                longest = max(width, height)
                if longest > max_edge:
                    scale = max_edge / float(longest)
                    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=90, optimize=True)
                base64_data = base64.b64encode(buf.getvalue()).decode("utf-8")
                return base64_data, "image/jpeg"

        # Default: preserve original bytes.
        with open(image_path, "rb") as image_file:
            base64_data = base64.b64encode(image_file.read()).decode("utf-8")
            return base64_data, mime_type
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None, None


def send_generate_request(
    messages,
    server_url=None,
    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    api_key=None,
    max_tokens=4096,
):
    """
    Sends a request to the OpenAI-compatible API endpoint using the OpenAI client library.

    Args:
        server_url (str): The base URL of the server, e.g. "http://127.0.0.1:8000"
        messages (list): A list of message dicts, each containing role and content.
        model (str): The model to use for generation (default: "llama-4")
        max_tokens (int): Maximum number of tokens to generate (default: 4096)

    Returns:
        str: The generated response text from the server.
    """
    # Process messages to convert image paths to base64
    image_detail = os.environ.get("SAM3_IMAGE_DETAIL", "low")
    try:
        image_max_edge = int(os.environ.get("SAM3_AGENT_IMAGE_MAX_EDGE", "896"))
    except ValueError:
        image_max_edge = 896

    processed_messages = []
    for message in messages:
        processed_message = message.copy()
        if message["role"] == "user" and "content" in message:
            processed_content = []
            for c in message["content"]:
                if isinstance(c, dict) and c.get("type") == "image":
                    # Convert image path to base64 format
                    image_path = c["image"]

                    print("image_path", image_path)
                    new_image_path = image_path.replace(
                        "?", "%3F"
                    )  # Escape ? in the path

                    # Read the image file and convert to base64
                    try:
                        base64_image, mime_type = get_image_base64_and_mime(
                            new_image_path,
                            max_edge=image_max_edge,
                        )
                        if base64_image is None:
                            print(
                                f"Warning: Could not convert image to base64: {new_image_path}"
                            )
                            continue

                        # Create the proper image_url structure with base64 data
                        processed_content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}",
                                    "detail": image_detail,
                                },
                            }
                        )

                    except FileNotFoundError:
                        print(f"Warning: Image file not found: {new_image_path}")
                        continue
                    except Exception as e:
                        print(f"Warning: Error processing image {new_image_path}: {e}")
                        continue
                else:
                    processed_content.append(c)

            processed_message["content"] = processed_content
        processed_messages.append(processed_message)

    # Create OpenAI client with custom base URL
    client = OpenAI(api_key=api_key, base_url=server_url)

    def _next_token_budget_from_error(error_text: str, current_budget: int) -> Optional[int]:
        """
        Parse vLLM/OpenAI-compatible context-window errors and compute a safer retry budget.
        Expected fragment:
          "You passed 4097 input tokens and requested 4096 output tokens... context length is only 8192..."
        """
        if "context length" not in error_text:
            return None

        # Always back off aggressively on context errors.
        fallback_budget = max(current_budget // 2, 64)

        input_match = re.search(r"passed\s+(\d+)\s+input tokens", error_text)
        context_match = re.search(r"context length is only\s+(\d+)\s+tokens", error_text)
        if not input_match or not context_match:
            if fallback_budget >= current_budget:
                return None
            return fallback_budget

        input_tokens = int(input_match.group(1))
        context_tokens = int(context_match.group(1))
        parsed_budget = max(context_tokens - input_tokens - 64, 64)
        next_budget = min(parsed_budget, fallback_budget)
        if next_budget >= current_budget:
            return None
        return next_budget

    budget = max_tokens
    # Retry a few times only for context-budget errors.
    for _attempt in range(6):
        try:
            print(f"🔍 Calling model {model}...")
            response = client.chat.completions.create(
                model=model,
                messages=processed_messages,
                max_completion_tokens=budget,
                n=1,
            )

            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content

            print(f"Unexpected response format: {response}")
            return None

        except Exception as e:
            error_text = str(e)
            next_budget = _next_token_budget_from_error(error_text, budget)
            if next_budget is None:
                print(f"Request failed: {e}")
                return None
            print(
                "Request exceeded context budget. "
                f"Retrying with max_completion_tokens={next_budget}."
            )
            budget = next_budget

    print("Request failed after retries due to repeated context budget errors.")
    return None


def send_direct_request(
    llm: Any,
    messages: list[dict[str, Any]],
    sampling_params: Any,
) -> Optional[str]:
    """
    Run inference on a vLLM model instance directly without using a server.

    Args:
        llm: Initialized vLLM LLM instance (passed from external initialization)
        messages: List of message dicts with role and content (OpenAI format)
        sampling_params: vLLM SamplingParams instance (initialized externally)

    Returns:
        str: Generated response text, or None if inference fails
    """
    try:
        # Process messages to handle images (convert to base64 if needed)
        processed_messages = []
        for message in messages:
            processed_message = message.copy()
            if message["role"] == "user" and "content" in message:
                processed_content = []
                for c in message["content"]:
                    if isinstance(c, dict) and c.get("type") == "image":
                        # Convert image path to base64 format
                        image_path = c["image"]
                        new_image_path = image_path.replace("?", "%3F")

                        try:
                            base64_image, mime_type = get_image_base64_and_mime(
                                new_image_path
                            )
                            if base64_image is None:
                                print(
                                    f"Warning: Could not convert image: {new_image_path}"
                                )
                                continue

                            # vLLM expects image_url format
                            processed_content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{base64_image}"
                                    },
                                }
                            )
                        except Exception as e:
                            print(
                                f"Warning: Error processing image {new_image_path}: {e}"
                            )
                            continue
                    else:
                        processed_content.append(c)

                processed_message["content"] = processed_content
            processed_messages.append(processed_message)

        print("🔍 Running direct inference with vLLM...")

        # Run inference using vLLM's chat interface
        outputs = llm.chat(
            messages=processed_messages,
            sampling_params=sampling_params,
        )

        # Extract the generated text from the first output
        if outputs and len(outputs) > 0:
            generated_text = outputs[0].outputs[0].text
            return generated_text
        else:
            print(f"Unexpected output format: {outputs}")
            return None

    except Exception as e:
        print(f"Direct inference failed: {e}")
        return None
