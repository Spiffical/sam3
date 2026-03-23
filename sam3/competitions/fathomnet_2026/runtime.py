from __future__ import annotations

import importlib
import json
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[3]

cv2 = None
Image = None
Sam3Processor = Any
agent_inference = None
build_sam3_image_model = None
remove_overlapping_masks = None
sam3_inference = None
send_generate_request_orig = None
visualize = None


def ensure_runtime_deps() -> None:
    global cv2, Image, Sam3Processor, agent_inference
    global build_sam3_image_model, remove_overlapping_masks
    global sam3_inference, send_generate_request_orig, visualize

    if (
        cv2 is not None
        and Image is not None
        and Sam3Processor is not Any
        and agent_inference is not None
    ):
        return

    missing: list[str] = []
    try:
        cv2 = importlib.import_module("cv2")
    except ImportError:
        missing.append("opencv-python")
    try:
        from PIL import Image as pil_image_module

        Image = pil_image_module
    except ImportError:
        missing.append("pillow")

    try:
        repo_root_str = str(REPO_ROOT)
        nibi_root_str = str(REPO_ROOT / "nibi_model_compare")
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)
        if nibi_root_str not in sys.path:
            sys.path.insert(0, nibi_root_str)
        from sam3.agent.agent_core import agent_inference as _agent_inference
        from sam3.agent.client_llm import (
            send_generate_request as _send_generate_request_orig,
        )
        from sam3.agent.client_sam3 import (
            remove_overlapping_masks as _remove_overlapping_masks,
            sam3_inference as _sam3_inference,
        )
        from sam3.agent.viz import visualize as _visualize
        from sam3.model.sam3_image_processor import Sam3Processor as _Sam3Processor
        from sam3.model_builder import (
            build_sam3_image_model as _build_sam3_image_model,
        )
    except ImportError as exc:
        missing.append(str(exc))
    else:
        Sam3Processor = _Sam3Processor
        agent_inference = _agent_inference
        build_sam3_image_model = _build_sam3_image_model
        remove_overlapping_masks = _remove_overlapping_masks
        sam3_inference = _sam3_inference
        send_generate_request_orig = _send_generate_request_orig
        visualize = _visualize

    if missing:
        raise RuntimeError(
            "Missing runtime dependencies: "
            + ", ".join(missing)
            + ". Activate the SAM3 runtime environment on Nibi before running."
        )


def find_bpe_path() -> str:
    env_path = os.environ.get("SAM3_BPE_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    candidates = [
        REPO_ROOT / "assets" / "bpe_simple_vocab_16e6.txt.gz",
        REPO_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz",
        Path("assets") / "bpe_simple_vocab_16e6.txt.gz",
        Path("sam3") / "assets" / "bpe_simple_vocab_16e6.txt.gz",
    ]
    for path in candidates:
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        "Could not find bpe_simple_vocab_16e6.txt.gz. "
        "Set SAM3_BPE_PATH or run from a repo checkout with assets present."
    )


def default_api_key(explicit_api_key: str = "") -> str:
    return (
        explicit_api_key
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("VLLM_API_KEY")
        or "DUMMY_API_KEY"
    )


def configure_agent_environment(
    *,
    prompt_profile: str,
    image_detail: str,
    max_images_per_request: int,
    agent_image_max_edge: int,
    agent_image_min_edge: int,
) -> None:
    os.environ["SAM3_AGENT_PROMPT_PROFILE"] = str(prompt_profile)
    os.environ["SAM3_IMAGE_DETAIL"] = str(image_detail)
    os.environ["SAM3_MAX_IMAGES_PER_REQUEST"] = str(max(1, int(max_images_per_request)))
    os.environ["SAM3_AGENT_IMAGE_MAX_EDGE"] = str(max(128, int(agent_image_max_edge)))
    os.environ["SAM3_AGENT_IMAGE_MIN_EDGE"] = str(max(128, int(agent_image_min_edge)))


def build_send_request(
    *,
    server_url: str,
    model: str,
    api_key: str,
    max_completion_tokens: int,
) -> Callable[[list[dict[str, Any]]], str | None]:
    ensure_runtime_deps()
    send_req_base = partial(
        send_generate_request_orig,
        server_url=server_url,
        model=model,
        api_key=api_key,
    )
    return partial(send_req_base, max_tokens=int(max_completion_tokens))


class LocalSam3Service:
    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def call_service(
        self,
        image_path: str,
        text_prompt: str,
        output_folder_path: str | None = None,
    ) -> str:
        ensure_runtime_deps()
        if output_folder_path is None:
            raise ValueError("output_folder_path is required for LocalSam3Service")
        os.makedirs(output_folder_path, exist_ok=True)

        outputs = sam3_inference(self.processor, image_path, text_prompt)
        outputs = remove_overlapping_masks(outputs)

        safe_prompt = text_prompt.replace("/", "_").replace(" ", "_")
        out_name = f"{Path(image_path).stem}_{safe_prompt}"
        output_image_path = os.path.join(output_folder_path, f"{out_name}.png")
        output_json_path = os.path.join(output_folder_path, f"{out_name}.json")

        outputs = {
            "original_image_path": image_path,
            "output_image_path": output_image_path,
            **outputs,
        }

        if "pred_scores" in outputs and outputs["pred_scores"]:
            order = sorted(
                range(len(outputs["pred_scores"])),
                key=lambda i: outputs["pred_scores"][i],
                reverse=True,
            )
            outputs["pred_scores"] = [outputs["pred_scores"][i] for i in order]
            outputs["pred_boxes"] = [outputs["pred_boxes"][i] for i in order]
            outputs["pred_masks"] = [outputs["pred_masks"][i] for i in order]

        valid_masks: list[Any] = []
        valid_boxes: list[Any] = []
        valid_scores: list[Any] = []
        for index, rle in enumerate(outputs.get("pred_masks", [])):
            if len(rle) > 4:
                valid_masks.append(rle)
                valid_boxes.append(outputs["pred_boxes"][index])
                valid_scores.append(outputs["pred_scores"][index])
        outputs["pred_masks"] = valid_masks
        outputs["pred_boxes"] = valid_boxes
        outputs["pred_scores"] = valid_scores

        with open(output_json_path, "w", encoding="utf-8") as handle:
            json.dump(outputs, handle, indent=2)
            handle.write("\n")
        visualize(outputs).save(output_image_path)
        return output_json_path
