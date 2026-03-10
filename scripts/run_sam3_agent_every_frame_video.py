#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import time
from functools import partial
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = "small creatures"
DEFAULT_PROMPT_PROFILE = "underwater"
DEFAULT_CODEC = "mp4v"

cv2 = None
np = None
Image = None
torch = None
Sam3Processor = Any
build_sam3_image_model = None
agent_inference = None
send_generate_request_orig = None
sam3_inference = None
remove_overlapping_masks = None
visualize = None


def ensure_runtime_deps() -> None:
    global cv2, np, Image, torch
    global Sam3Processor, build_sam3_image_model
    global agent_inference, send_generate_request_orig
    global sam3_inference, remove_overlapping_masks, visualize
    if cv2 is not None and np is not None and Image is not None and torch is not None:
        return

    missing: list[str] = []
    try:
        cv2 = importlib.import_module("cv2")
    except ImportError:
        missing.append("opencv-python")
    try:
        np = importlib.import_module("numpy")
    except ImportError:
        missing.append("numpy")
    try:
        import torch as torch_mod

        torch = torch_mod
    except ImportError:
        missing.append("torch")
    try:
        from PIL import Image as pil_image_module

        Image = pil_image_module
    except ImportError:
        missing.append("pillow")
    try:
        from sam3.model.sam3_image_processor import Sam3Processor as _Sam3Processor
        from sam3.model_builder import (
            build_sam3_image_model as _build_sam3_image_model,
        )
        from sam3.agent.agent_core import agent_inference as _agent_inference
        from sam3.agent.client_llm import (
            send_generate_request as _send_generate_request_orig,
        )
        from sam3.agent.client_sam3 import (
            remove_overlapping_masks as _remove_overlapping_masks,
            sam3_inference as _sam3_inference,
        )
        from sam3.agent.viz import visualize as _visualize
    except ImportError as exc:
        missing.append(str(exc))
    else:
        Sam3Processor = _Sam3Processor
        build_sam3_image_model = _build_sam3_image_model
        agent_inference = _agent_inference
        send_generate_request_orig = _send_generate_request_orig
        remove_overlapping_masks = _remove_overlapping_masks
        sam3_inference = _sam3_inference
        visualize = _visualize

    if missing:
        raise RuntimeError(
            "Missing runtime dependencies: "
            + ", ".join(missing)
            + ". Activate the SAM3 environment on DRAC before running this script."
        )


def default_device() -> str:
    try:
        import torch as torch_mod

        return "cuda" if torch_mod.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


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

    try:
        import importlib_resources

        resource_path = importlib_resources.files("sam3").joinpath(
            "assets/bpe_simple_vocab_16e6.txt.gz"
        )
        if resource_path.is_file():
            return str(resource_path)
    except Exception:
        pass

    raise FileNotFoundError(
        "Could not find bpe_simple_vocab_16e6.txt.gz. "
        "Set SAM3_BPE_PATH or run from a repo checkout with assets present."
    )


def log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


class ProgressReporter:
    def __init__(self, total: int, desc: str) -> None:
        self.total = max(0, int(total))
        self.desc = desc
        self.start_time = time.time()
        self.current = 0
        self._bar = (
            tqdm(total=self.total, desc=self.desc, unit="frame", dynamic_ncols=True)
            if tqdm is not None
            else None
        )

    def update(self, count: int = 1, *, postfix: dict[str, Any] | None = None) -> None:
        self.current += int(count)
        if self._bar is not None:
            self._bar.update(count)
            if postfix:
                self._bar.set_postfix(postfix, refresh=False)
            return

        should_print = (
            self.current == self.total
            or self.current == 1
            or self.current % 5 == 0
        )
        if should_print:
            elapsed = max(1e-6, time.time() - self.start_time)
            fps = self.current / elapsed
            suffix = ""
            if postfix:
                suffix = " | " + ", ".join(f"{k}={v}" for k, v in postfix.items())
            print(
                f"\r{self.desc}: {self.current}/{self.total} ({fps:.2f} frames/s){suffix}",
                end="" if self.current < self.total else "\n",
                flush=True,
            )

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the SAM3 agent loop on every frame of a video and write an overlay video."
        )
    )
    parser.add_argument("video_path", help="Input video path.")
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=f"Initial user prompt for the agent. Default: {DEFAULT_PROMPT!r}",
    )
    parser.add_argument(
        "--prompt-profile",
        default=DEFAULT_PROMPT_PROFILE,
        help=f"SAM3 agent prompt profile. Default: {DEFAULT_PROMPT_PROFILE!r}",
    )
    parser.add_argument(
        "--server-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
        help="OpenAI-compatible server URL for the MLLM.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "Qwen/Qwen3.5-27B"),
        help="OpenAI-compatible model ID for the MLLM.",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Optional API key. Defaults to OPENAI_API_KEY or VLLM_API_KEY if set.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=1024,
        help="Maximum tokens per MLLM completion. Default: 1024",
    )
    parser.add_argument(
        "--max-generations",
        type=int,
        default=10,
        help="Maximum agent generations per frame. Default: 10",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Default: <video_stem>_sam3_agent_every_frame",
    )
    parser.add_argument(
        "--output-video-path",
        default="",
        help="Optional explicit output video path. Default: <output-dir>/overlay.mp4",
    )
    parser.add_argument(
        "--summary-path",
        default="",
        help="Optional explicit summary JSON path. Default: <output-dir>/summary.json",
    )
    parser.add_argument(
        "--frame-results-path",
        default="",
        help="Optional JSONL file to write per-frame agent results.",
    )
    parser.add_argument(
        "--codec",
        default=DEFAULT_CODEC,
        help=f"OpenCV fourcc codec. Default: {DEFAULT_CODEC}",
    )
    parser.add_argument(
        "--device",
        default=default_device(),
        help="Model device. Default: cuda if available, otherwise cpu.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="",
        help="Optional local SAM3 checkpoint path.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.40,
        help="SAM3 image processor confidence threshold. Default: 0.40",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable model compilation when building the image model.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional frame limit for quick tests. Default: 0 (all frames).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Keep full per-frame agent debug outputs.",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Preserve per-frame input images and agent artifact folders.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Write the raw frame on agent failure and continue processing.",
    )
    parser.add_argument(
        "--system-prompt-path",
        default="",
        help="Optional override for the base SAM3 agent system prompt.",
    )
    parser.add_argument(
        "--iterative-system-prompt-path",
        default="",
        help="Optional override for the iterative mask-checking system prompt.",
    )
    return parser.parse_args()


def make_output_paths(args: argparse.Namespace) -> tuple[str, str, str]:
    video_path = Path(args.video_path).resolve()
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = video_path.with_name(f"{video_path.stem}_sam3_agent_every_frame")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_video_path = (
        Path(args.output_video_path).resolve()
        if args.output_video_path
        else output_dir / "overlay.mp4"
    )
    summary_path = (
        Path(args.summary_path).resolve()
        if args.summary_path
        else output_dir / "summary.json"
    )
    frame_results_path = (
        Path(args.frame_results_path).resolve()
        if args.frame_results_path
        else output_dir / "frame_results.jsonl"
    )
    return str(output_video_path), str(summary_path), str(frame_results_path)


class LocalSam3Service:
    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def call_service(
        self,
        image_path: str,
        text_prompt: str,
        output_folder_path: str | None = None,
    ) -> str:
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
        for i, rle in enumerate(outputs.get("pred_masks", [])):
            if len(rle) > 4:
                valid_masks.append(rle)
                valid_boxes.append(outputs["pred_boxes"][i])
                valid_scores.append(outputs["pred_scores"][i])
        outputs["pred_masks"] = valid_masks
        outputs["pred_boxes"] = valid_boxes
        outputs["pred_scores"] = valid_scores

        with open(output_json_path, "w", encoding="utf-8") as handle:
            json.dump(outputs, handle, indent=2)
            handle.write("\n")
        visualize(outputs).save(output_image_path)
        return output_json_path


def history_segment_prompts(history: list[dict[str, Any]]) -> list[str]:
    prompts: list[str] = []
    seen: set[str] = set()
    for message in history:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "text":
                continue
            text = str(item.get("text", ""))
            if "segment_phrase" not in text or "<tool>" not in text:
                continue
            start = text.find("<tool>")
            end = text.find("</tool>")
            if start < 0 or end < 0:
                continue
            payload = text[start + len("<tool>") : end].strip()
            try:
                parsed = json.loads(payload)
            except Exception:
                continue
            if parsed.get("name") != "segment_phrase":
                continue
            params = parsed.get("parameters", {})
            prompt = str(params.get("text_prompt", "")).strip()
            if prompt and prompt not in seen:
                prompts.append(prompt)
                seen.add(prompt)
    return prompts


def pil_to_bgr(image: Any) -> Any:
    rgb = image.convert("RGB")
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)


def write_frame_result(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload) + "\n")
    handle.flush()


def main() -> int:
    args = parse_args()
    ensure_runtime_deps()

    output_video_path, summary_path, frame_results_path = make_output_paths(args)
    output_dir = str(Path(output_video_path).resolve().parent)
    video_path = str(Path(args.video_path).resolve())
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    os.environ["SAM3_AGENT_PROMPT_PROFILE"] = str(args.prompt_profile)
    if args.system_prompt_path:
        os.environ["SAM3_SYSTEM_PROMPT_PATH"] = str(
            Path(args.system_prompt_path).resolve()
        )
    if args.iterative_system_prompt_path:
        os.environ["SAM3_ITERATIVE_SYSTEM_PROMPT_PATH"] = str(
            Path(args.iterative_system_prompt_path).resolve()
        )

    log("Building SAM3 image model for the local tool loop...")
    bpe_path = find_bpe_path()
    image_model = build_sam3_image_model(
        bpe_path=bpe_path,
        device=str(args.device),
        checkpoint_path=(args.checkpoint_path or None),
        compile=bool(args.compile),
    )
    image_processor = Sam3Processor(
        image_model, confidence_threshold=float(args.confidence_threshold)
    )
    local_service = LocalSam3Service(image_processor)

    api_key = (
        args.api_key
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("VLLM_API_KEY")
        or "DUMMY_API_KEY"
    )
    send_req = partial(
        send_generate_request_orig,
        server_url=args.server_url,
        model=args.model,
        api_key=api_key,
        max_tokens=int(args.max_completion_tokens),
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    max_frames = int(args.max_frames) if int(args.max_frames) > 0 else total_frames
    if max_frames <= 0:
        raise RuntimeError("Video appears to contain no frames.")

    writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*str(args.codec)),
        fps,
        (frame_w, frame_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_video_path}")

    frame_inputs_dir = os.path.join(output_dir, "frame_inputs")
    frame_artifacts_dir = os.path.join(output_dir, "agent_frames")
    if args.keep_artifacts or args.debug:
        os.makedirs(frame_inputs_dir, exist_ok=True)
        os.makedirs(frame_artifacts_dir, exist_ok=True)

    log(
        "Starting frame-by-frame SAM3 agent loop "
        f"for {max_frames} frame(s) with prompt={args.prompt!r}, "
        f"profile={args.prompt_profile!r}, max_generations={args.max_generations}"
    )

    progress = ProgressReporter(total=max_frames, desc="SAM3 agent")
    start_time = time.time()
    processed_frames = 0
    total_masks = 0
    frames_with_masks = 0
    error_count = 0

    with open(frame_results_path, "w", encoding="utf-8") as frame_results_handle:
        try:
            for frame_index in range(max_frames):
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                frame_name = f"frame_{frame_index:06d}.jpg"
                if args.keep_artifacts or args.debug:
                    frame_path = os.path.join(frame_inputs_dir, frame_name)
                else:
                    frame_path = os.path.join(output_dir, "_current_frame.jpg")
                if not cv2.imwrite(frame_path, frame_bgr):
                    raise RuntimeError(f"Could not write temporary frame image: {frame_path}")

                frame_agent_dir = os.path.join(
                    frame_artifacts_dir if (args.keep_artifacts or args.debug) else output_dir,
                    f"frame_{frame_index:06d}",
                )
                frame_start = time.time()
                frame_error: str | None = None
                history: list[dict[str, Any]] = []
                final_outputs: dict[str, Any] = {
                    "pred_masks": [],
                    "pred_scores": [],
                }

                try:
                    history, final_outputs, rendered_img = agent_inference(
                        img_path=frame_path,
                        initial_text_prompt=args.prompt,
                        debug=bool(args.debug),
                        send_generate_request=send_req,
                        call_sam_service=local_service.call_service,
                        max_generations=int(args.max_generations),
                        output_dir=frame_agent_dir,
                    )
                    overlaid_frame = pil_to_bgr(rendered_img)
                except Exception as exc:
                    frame_error = f"{type(exc).__name__}: {str(exc).strip() or repr(exc)}"
                    error_count += 1
                    if not args.continue_on_error:
                        raise
                    overlaid_frame = frame_bgr.copy()
                finally:
                    if not (args.keep_artifacts or args.debug):
                        try:
                            if os.path.isfile(frame_path):
                                os.remove(frame_path)
                        except Exception:
                            pass
                        try:
                            if os.path.isdir(frame_agent_dir):
                                shutil.rmtree(frame_agent_dir)
                        except Exception:
                            pass

                writer.write(overlaid_frame)

                num_masks = len(final_outputs.get("pred_masks", []))
                segment_prompts = history_segment_prompts(history)
                processed_frames += 1
                total_masks += int(num_masks)
                if num_masks > 0:
                    frames_with_masks += 1

                per_frame_payload = {
                    "frame_index": int(frame_index),
                    "num_masks": int(num_masks),
                    "segment_prompts": segment_prompts,
                    "history_len": int(len(history)),
                    "frame_runtime_sec": float(time.time() - frame_start),
                    "error": frame_error,
                }
                write_frame_result(frame_results_handle, per_frame_payload)

                avg_masks = total_masks / float(max(1, processed_frames))
                progress.update(
                    1,
                    postfix={
                        "masks": num_masks,
                        "avg_masks": f"{avg_masks:.2f}",
                        "errors": error_count,
                    },
                )
        finally:
            progress.close()
            cap.release()
            writer.release()

    runtime_sec = time.time() - start_time
    summary = {
        "video_path": video_path,
        "output_dir": output_dir,
        "output_video_path": output_video_path,
        "summary_path": summary_path,
        "frame_results_path": frame_results_path,
        "frame_inputs_dir": frame_inputs_dir if (args.keep_artifacts or args.debug) else "",
        "frame_artifacts_dir": (
            frame_artifacts_dir if (args.keep_artifacts or args.debug) else ""
        ),
        "prompt": args.prompt,
        "prompt_profile": args.prompt_profile,
        "server_url": args.server_url,
        "model": args.model,
        "device": args.device,
        "max_generations": int(args.max_generations),
        "max_completion_tokens": int(args.max_completion_tokens),
        "fps": float(fps),
        "frame_size_hw": [int(frame_h), int(frame_w)],
        "processed_frames": int(processed_frames),
        "frames_with_masks": int(frames_with_masks),
        "total_masks": int(total_masks),
        "avg_masks_per_frame": float(total_masks) / float(max(1, processed_frames)),
        "error_count": int(error_count),
        "keep_artifacts": bool(args.keep_artifacts),
        "debug": bool(args.debug),
        "continue_on_error": bool(args.continue_on_error),
        "runtime_sec": float(runtime_sec),
        "throughput_fps": float(processed_frames) / float(max(1e-6, runtime_sec)),
        "finished_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    log(f"Wrote overlay video: {output_video_path}")
    log(f"Wrote summary JSON: {summary_path}")
    log(f"Wrote per-frame results: {frame_results_path}")
    if args.keep_artifacts or args.debug:
        log(f"Preserved frame artifacts under: {frame_artifacts_dir}")
    log(
        "Done. Note: the output video is re-encoded via OpenCV and does not preserve audio."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("Interrupted.")
        raise SystemExit(130)
