
from __future__ import annotations
import base64
import json
import tempfile
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from sam3.visualization_utils import render_masklet_frame

_SESSION_CACHE = {}

MAX_DISPLAY_WIDTH = 800

def _get_session_data(session_id: str) -> Dict:
    if not session_id: return {}
    return _SESSION_CACHE.get(session_id, {})

def _default_state() -> Dict:
    return {
        "session_id": None,
        "video_path": None,
        "temp_dir": None,
        "display_dir": None,
        # "frames": [], # We don't store frames in state anymore, logic is in SESSION_CACHE
        # "display_frames": [],
        "display_size": (0, 0),
        "frame_outputs": {},
        "clicks": {},
        "current_frame": 0,
        "frame_size": None,
        "status": "Upload a video or JPEG folder to get going.",
        "click_history": [],
        "obj_metadata": {},
    }

def _ensure_display_dir(state_dict: Dict) -> Path:
    display_dir = state_dict.get("display_dir")
    if display_dir is None:
        # Use a local tmp directory to avoid permission issues with /tmp
        base_tmp = Path.cwd() / "tmp"
        base_tmp.mkdir(parents=True, exist_ok=True)
        display_dir = Path(tempfile.mkdtemp(prefix="sam3_display_frames_", dir=base_tmp))
        state_dict["display_dir"] = str(display_dir)
    else:
        display_dir = Path(display_dir)
    display_dir.mkdir(parents=True, exist_ok=True)
    return display_dir

def _ensure_display_dimensions(state_dict: Dict):
    size = state_dict.get("display_size")
    if size and size[0] > 0 and size[1] > 0:
        return
    sid = state_dict.get("session_id")
    # Note: Frames removed from state_dict, must fetch from session cache
    session_data = _get_session_data(sid)
    frames = session_data.get("frames")
    
    if not frames:
        state_dict["display_size"] = (0, 0)
        return
    orig_h, orig_w = frames[0].shape[:2]
    target_w = min(MAX_DISPLAY_WIDTH, orig_w)
    scale = target_w / orig_w if orig_w else 1.0
    target_h = max(1, int(round(orig_h * scale)))
    state_dict["display_size"] = (int(target_w), int(target_h))

def _refresh_display_frames(
    state_dict: Dict, frame_indices: Optional[List[int]] = None
) -> List[int]:
    sid = state_dict.get("session_id")
    session_data = _get_session_data(sid)
    frames = session_data.get("frames")
    
    if not frames:
        state_dict["display_size"] = (0, 0)
        return []

    _ensure_display_dimensions(state_dict)
    disp_w, disp_h = state_dict.get("display_size", (0, 0))
    
    if frames is None:
        return []
        
    display_frames = session_data.get("display_frames")
    if display_frames is None or len(display_frames) != len(frames):
        session_data["display_frames"] = [""] * len(frames)
        display_frames = session_data["display_frames"]
        frame_indices = None

    target_indices = frame_indices or list(range(len(frames)))
    print(
        f"[DEBUG] Refreshing {len(target_indices)} frames "
        f"(total={len(frames)}, updated={target_indices[:10]})"
    )
    
    current_frame_idx = state_dict.get("current_frame", -1)
    preview_mode = state_dict.get("preview_mode", False)
    preview_outputs = state_dict.get("preview_outputs")

    for idx in target_indices:
        if idx < 0 or idx >= len(frames):
            continue
        rgb_frame = frames[idx].copy()
        
        # Determine which outputs to render
        outputs = None
        if preview_mode and idx == current_frame_idx and preview_outputs:
             outputs = preview_outputs
        else:
             outputs = state_dict["frame_outputs"].get(idx)
             
        if outputs:
            rgb_frame = render_masklet_frame(rgb_frame, outputs, frame_idx=idx)
            
            # --- Render Class Labels ---
            obj_metadata = state_dict.get("obj_metadata", {})
            if obj_metadata:
                 boxes = outputs.get("out_boxes_xywh", [])
                 ids = outputs.get("out_obj_ids", [])
                 for obj_id, (x, y, w, h) in zip(ids, boxes):
                      # Convert to int
                      obj_id = int(obj_id)
                      label_text = obj_metadata.get(obj_id)
                      # Also default to just ID if requested? 
                      # The user specifically asked for "taxonomic classification".
                      
                      if label_text:
                           text = f"{label_text} ({obj_id})"
                           
                           # Draw text
                           # x, y are top-left of box
                           pt = (int(x), int(y) - 5)
                           
                           # Draw black background for text
                           (fw, fh), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                           cv2.rectangle(
                               rgb_frame, 
                               (pt[0], pt[1] - fh - 2), 
                               (pt[0] + fw, pt[1] + baseline - 2), 
                               (0, 0, 0), 
                               -1
                           )
                           
                           cv2.putText(
                               rgb_frame,
                               text,
                               pt,
                               cv2.FONT_HERSHEY_SIMPLEX,
                               0.6,
                               (255, 255, 255),
                               2,
                               cv2.LINE_AA
                           )
            # ---------------------------
        
        resized = cv2.resize(
            rgb_frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR
        )
        success, buffer = cv2.imencode(
            ".jpg",
            cv2.cvtColor(resized, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 75],
        )
        b64_data = base64.b64encode(buffer).decode("utf-8")
        display_frames[idx] = f"data:image/jpeg;base64,{b64_data}"
    return target_indices


def _build_frames_payload(
    state_dict: Dict,
    updated_indices: Optional[List[int]] = None,
) -> str:
    session_data = _get_session_data(state_dict.get("session_id"))
    display_frames = session_data.get("display_frames") or []
    disp_w, disp_h = state_dict.get("display_size", (0, 0))
    display_dir = state_dict.get("display_dir")
    
    # IMPORTANT: Start session sends "full" payload. We must check if display_frames is available.
    if not display_frames:
         # Fallback if display_frames is empty but frames exist? 
         # _refresh_display_frames should have populated it.
         if not session_data.get("frames"):
             return json.dumps({"type": "clear"})
    
    # Convert to absolute path string for Gradio serving
    abs_display_dir = str(Path(display_dir).resolve()) if display_dir else ""
    payload: Dict[str, Any] = {"width": disp_w, "height": disp_h, "dir": abs_display_dir}
    if (
        updated_indices is None
        or len(updated_indices) == len(display_frames)
        or not updated_indices
    ):
        payload["type"] = "full"
        payload["frames"] = display_frames
    else:
        payload["type"] = "patch"
        payload["patches"] = [
            {"index": idx, "frame": display_frames[idx]}
            for idx in updated_indices
            if 0 <= idx < len(display_frames)
        ]
    payload_json = json.dumps(payload)
    # print(f"[DEBUG] Payload len: {len(payload_json)}")
    return payload_json

def _detach_outputs(raw: Dict) -> Dict:
    clean = {
        "out_boxes_xywh": [],
        "out_probs": [],
        "out_obj_ids": [],
        "out_binary_masks": [],
    }
    for key in ("out_boxes_xywh", "out_probs", "out_obj_ids"):
        val = raw[key]
        if isinstance(val, torch.Tensor):
            clean[key] = val.detach().cpu().numpy().tolist()
        else:
            clean[key] = (
                val.tolist() if hasattr(val, "tolist") else list(val)  # type: ignore[arg-type]
            )
    for mask in raw["out_binary_masks"]:
        if isinstance(mask, torch.Tensor):
            clean["out_binary_masks"].append(mask.detach().cpu().numpy())
        elif isinstance(mask, np.ndarray):
            clean["out_binary_masks"].append(mask)
        else:
            clean["out_binary_masks"].append(np.array(mask))
    return clean

def _render_frame(state: Dict) -> np.ndarray | None:
    # Not really used for display anymore (we use display_frames), but kept for reference if needed
    if not state.get("frames"):
        # We need to get frames from session cache
        sid = state.get("session_id")
        frames = _get_session_data(sid).get("frames")
        if not frames:
            return None
    else:
        frames = state["frames"] # Logic removed in start_session but function signature remains

    # This function seems to be unused in the main flow shown in the original file
    # _refresh_display_frames does the rendering.
    # But let's keep it if we need full res render for some reason.
    pass

def _format_object_summary(state: Dict) -> str:
    outputs = state["frame_outputs"].get(state["current_frame"])
    if not outputs:
        return "No tracked objects on this frame yet."
    obj_ids = [int(obj) for obj in outputs["out_obj_ids"]]
    return f"Tracked IDs on this frame: {sorted(set(obj_ids))}"


def _format_click_summary(state: Dict) -> str:
    frame_clicks = state["clicks"].get(state["current_frame"], {})
    if not frame_clicks:
        return "No clicks logged on this frame."
    lines: List[str] = []
    for obj_id in sorted(frame_clicks.keys()):
        for idx, (x, y, label) in enumerate(frame_clicks[obj_id], start=1):
            lines.append(
                f"obj {obj_id} | click {idx} | ({int(x)}, {int(y)}) | {'+' if label == 1 else '-'}"
            )
    return "\n".join(lines)
